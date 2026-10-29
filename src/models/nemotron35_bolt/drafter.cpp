#include "src/models/nemotron35_bolt/drafter.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/device_model.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/drafter_kernels.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/kernels.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/mmq/nb_mmq.h"
#include "src/models/nemotron35_bolt/weights.hpp"

namespace gufo::models::nemotron35_bolt {
namespace {

/// Projected context rows kept for the LiLiCorr anchor, by position.
constexpr int kAnchorRows = 16;

template<class T>
struct Buffer {
  T* data{nullptr};
  std::size_t count{0};

  Buffer() = default;
  Buffer(Buffer&& other) noexcept
      : data(std::exchange(other.data, nullptr)), count(other.count) {}
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() {
    if (data != nullptr)
      static_cast<void>(hipFree(data));
  }
  bool Allocate(std::size_t n) {
    count = n;
    return hipMalloc(&data, std::max<std::size_t>(n, 1) * sizeof(T)) ==
           hipSuccess;
  }
};

}  // namespace

Drafter::~Drafter() = default;

std::unique_ptr<Drafter> Drafter::Load(const std::string& path,
                                       const Config& target,
                                       std::string* error_msg) {
  auto fail = [&](std::string message) -> std::unique_ptr<Drafter> {
    if (error_msg != nullptr)
      *error_msg = std::move(message);
    return nullptr;
  };
  std::unique_ptr<Drafter> d(new Drafter());
  std::string error;
  d->reader_ = core::GgufReader::OpenFile(path, &error);
  if (!d->reader_)
    return fail("open " + path + ": " + error);
  auto weights = DrafterWeights::Bind(*d->reader_, target, &error);
  if (!weights)
    return fail(path + ": " + error);
  d->weights_ = std::make_unique<DrafterWeights>(std::move(*weights));
  d->device_ = rocm::DeviceDrafter::Upload(*d->weights_, *d->reader_, &error);
  if (!d->device_)
    return fail(error);
  return d;
}

const DrafterConfig& Drafter::config() const noexcept {
  return weights_->config;
}

std::size_t Drafter::ResidentBytes() const noexcept {
  return device_->bytes;
}

std::vector<std::int32_t> DraftLattice::Walk(std::uint32_t max_tokens) const {
  std::vector<std::int32_t> out;
  const std::uint32_t n = std::min(max_tokens, slots);
  std::uint32_t prev = 0;
  for (std::uint32_t s = 0; s < n; ++s) {
    const float* row =
        s == 0 ? scores.data()
               : scores.data() + top_k + ((s - 1) * top_k + prev) * top_k;
    prev = static_cast<std::uint32_t>(std::max_element(row, row + top_k) - row);
    out.push_back(candidates[s * top_k + prev]);
  }
  return out;
}

struct DraftContext::Buffers {
  // Rings over the trunk context, per drafter layer.
  std::vector<Buffer<__half>> k_ring, v_ring;
  Buffer<float> anchors;  ///< [kAnchorRows][hidden] projected context rows
  int ring{0};
  // Injection scratch, max_rows rows.
  Buffer<float> features;  ///< Host-injected features only.
  Buffer<float> context;   ///< fc + enc_norm output
  Buffer<__hip_bfloat16> context_bf16;
  Buffer<float> ck, cv;
  Buffer<int> first;  ///< InjectHost's position.
  // Block scratch.
  Buffer<std::int32_t> tokens;
  Buffer<int> pos0;
  Buffer<float> x, h, hc, dyn, q, k, v, ctx, o, gate, up, act, hidden, logits;
  // LiLiCorr scratch.
  Buffer<std::int32_t> cand;
  Buffer<float> values, lse, feat5, feat_up, feat, embd, token, pass, nodes,
      anchor, normed, qkv, att, proj, mlp, factor_in, factors, out_head,
      in_head, anchor_head, scores;
};

DraftContext::~DraftContext() = default;

bool DraftContext::Fail(std::string* error_msg, std::string message) const {
  if (error_msg != nullptr)
    *error_msg = std::move(message);
  return false;
}

std::unique_ptr<DraftContext> DraftContext::Create(
    const Drafter& drafter, const rocm::DeviceModel& trunk,
    std::uint32_t max_rows, hipStream_t stream, hipblasHandle_t blas,
    std::string* error_msg) {
  const DrafterConfig& c = drafter.config();
  std::unique_ptr<DraftContext> d(new DraftContext());
  d->drafter_ = &drafter;
  d->trunk_ = &trunk;
  d->stream_ = stream;
  d->blas_ = blas;
  d->max_rows_ = std::max<std::uint32_t>(max_rows, 1);
  d->b_ = std::make_unique<Buffers>();
  Buffers& b = *d->b_;
  const std::size_t H = c.hidden_size;
  const std::size_t R = d->max_rows_;
  const std::size_t T = c.block_size;
  const std::size_t S = c.Slots();
  const std::size_t N = S * c.lc_top_k;
  const std::size_t L = c.lc_hidden;
  const std::size_t vocab = trunk.output().rows;
  b.ring = static_cast<int>(c.sliding_window + 4 * c.block_size);
  bool ok = true;
  b.k_ring.resize(c.num_layers);
  b.v_ring.resize(c.num_layers);
  for (std::uint32_t il = 0; il < c.num_layers; ++il)
    ok = ok && b.k_ring[il].Allocate(b.ring * c.KvDim()) &&
         b.v_ring[il].Allocate(b.ring * c.KvDim());
  ok = ok && b.anchors.Allocate(kAnchorRows * H) && b.context.Allocate(R * H) &&
       b.context_bf16.Allocate(R * H) && b.ck.Allocate(R * c.KvDim()) &&
       b.cv.Allocate(R * c.KvDim()) && b.first.Allocate(1) &&
       b.tokens.Allocate(T) && b.pos0.Allocate(1) && b.x.Allocate(T * H) &&
       b.h.Allocate(T * H) && b.hc.Allocate(T * H) &&
       b.dyn.Allocate(T * c.ConvProjection()) && b.q.Allocate(T * c.QDim()) &&
       b.k.Allocate(T * c.KvDim()) && b.v.Allocate(T * c.KvDim()) &&
       b.ctx.Allocate(T * c.QDim()) && b.o.Allocate(T * H) &&
       b.gate.Allocate(T * c.ff_size) && b.up.Allocate(T * c.ff_size) &&
       b.act.Allocate(T * c.ff_size) && b.hidden.Allocate(T * H) &&
       b.logits.Allocate(S * vocab) && b.cand.Allocate(N) &&
       b.values.Allocate(N) && b.lse.Allocate(S) && b.feat5.Allocate(N * 5) &&
       b.feat_up.Allocate(N * L) && b.feat.Allocate(N * L) &&
       b.embd.Allocate(N * H) && b.token.Allocate(N * L) &&
       b.pass.Allocate(S * L) && b.nodes.Allocate(N * L) &&
       b.anchor.Allocate(L) && b.normed.Allocate(N * L) &&
       b.qkv.Allocate(N * 3 * L) && b.att.Allocate(N * L) &&
       b.proj.Allocate(N * L) && b.mlp.Allocate(N * c.lc_mlp) &&
       b.factor_in.Allocate(N * 3 * L) && b.factors.Allocate(N * L) &&
       b.out_head.Allocate(N * c.lc_factor) &&
       b.in_head.Allocate(N * c.lc_factor) &&
       b.anchor_head.Allocate(c.lc_factor) &&
       b.scores.Allocate(c.lc_top_k + (S - 1) * c.lc_top_k * c.lc_top_k);
  if (!ok) {
    d->Fail(error_msg, "could not allocate drafter memory");
    return nullptr;
  }
  // Mask tokens fill the block after its anchor.
  std::vector<std::int32_t> block(T, static_cast<std::int32_t>(c.mask_token));
  if (hipMemcpyAsync(b.tokens.data, block.data(), T * sizeof(std::int32_t),
                     hipMemcpyHostToDevice, stream) != hipSuccess ||
      hipStreamSynchronize(stream) != hipSuccess) {
    d->Fail(error_msg, "could not initialize the drafter block");
    return nullptr;
  }
  return d;
}

namespace {

/// Enqueues drafter work and keeps the first failure.
class Pass {
public:
  Pass(hipStream_t stream, hipblasHandle_t blas) : s_(stream), blas_(blas) {}

  void Ok(hipError_t e) {
    if (ok_ && e != hipSuccess) {
      ok_ = false;
      error_ = hipGetErrorString(e);
    }
  }
  /// NVFP4 or Q8_0 weights through the quantized kernels: out = x W^T.
  void Quant(const rocm::DeviceTensor& w, const float* x, float* out, int n,
             const char* what) {
    if (ok_ && nb_mmq_dense(static_cast<int>(w.type), w.data, x, out, w.rows, n,
                            w.cols, s_) != 0) {
      ok_ = false;
      error_ = std::string("quantized matrix product failed: ") + what;
    }
  }
  /// F32 weights: out = x W^T.
  void F32(const rocm::DeviceTensor& w, const float* x, float* out, int n) {
    const float one = 1.0F;
    const float zero = 0.0F;
    if (ok_ && hipblasSgemm(blas_, HIPBLAS_OP_T, HIPBLAS_OP_N, w.rows, n,
                            w.cols, &one, w.f32(), w.cols, x, w.cols, &zero,
                            out, w.rows) != HIPBLAS_STATUS_SUCCESS) {
      ok_ = false;
      error_ = "F32 GEMM failed";
    }
  }
  /// BF16 weights: a streaming kernel up to kGemvMaxTokens rows, hipBLAS
  /// with BF16 activations beyond.
  void Bf16(const rocm::DeviceTensor& w, const float* x, float* out, int n,
            __hip_bfloat16* x_bf16) {
    const auto* wb = static_cast<const __hip_bfloat16*>(w.data);
    if (n <= rocm::kGemvMaxTokens) {
      Ok(rocm::GemvBf16(wb, x, out, w.rows, w.cols, n, s_));
      return;
    }
    Ok(rocm::ToBf16(x, x_bf16, static_cast<std::size_t>(n) * w.cols, s_));
    const float one = 1.0F;
    const float zero = 0.0F;
    if (ok_ &&
        hipblasGemmEx(blas_, HIPBLAS_OP_T, HIPBLAS_OP_N, w.rows, n, w.cols,
                      &one, wb, HIP_R_16BF, w.cols, x_bf16, HIP_R_16BF, w.cols,
                      &zero, out, HIP_R_32F, w.rows, HIPBLAS_COMPUTE_32F,
                      HIPBLAS_GEMM_DEFAULT) != HIPBLAS_STATUS_SUCCESS) {
      ok_ = false;
      error_ = "BF16 GEMM failed";
    }
  }

  [[nodiscard]] bool ok() const { return ok_; }
  [[nodiscard]] const std::string& error() const { return error_; }
  [[nodiscard]] hipStream_t stream() const { return s_; }

private:
  hipStream_t s_;
  hipblasHandle_t blas_;
  bool ok_{true};
  std::string error_;
};

}  // namespace

bool DraftContext::Inject(const float* features, const int* first, int n,
                          std::string* error_msg) {
  if (n <= 0)
    return true;
  if (static_cast<std::uint32_t>(n) > max_rows_)
    return Fail(error_msg, "drafter injection exceeds its row budget");
  const DrafterConfig& c = drafter_->config();
  const rocm::DeviceDrafter& w = drafter_->device();
  Buffers& b = *b_;
  const int H = static_cast<int>(c.hidden_size);
  const int kv = static_cast<int>(c.KvDim());
  Pass p(stream_, blas_);
  p.Quant(w.fc, features, b.context.data, n, "fc");
  p.Ok(rocm::ScaleBias(b.context.data, w.fc.scale, nullptr, n, H,
                       rocm::Activation::kNone, stream_));
  p.Ok(rocm::RmsNorm(b.context.data, w.enc_norm.f32(), b.context.data, n, H,
                     c.rms_eps, stream_));
  p.Ok(rocm::RowsToRing(b.context.data, b.anchors.data, first, n, H,
                        kAnchorRows, stream_));
  for (std::uint32_t il = 0; p.ok() && il < c.num_layers; ++il) {
    const rocm::DeviceDrafterLayer& l = w.layers[il];
    p.Bf16(l.attn_k, b.context.data, b.ck.data, n, b.context_bf16.data);
    p.Bf16(l.attn_v, b.context.data, b.cv.data, n, b.context_bf16.data);
    p.Ok(rocm::RmsNorm(b.ck.data, l.attn_k_norm.f32(), b.ck.data,
                       n * static_cast<int>(c.num_kv_heads),
                       static_cast<int>(c.head_dim), c.rms_eps, stream_));
    p.Ok(rocm::RopeNeox(b.ck.data, first, n, static_cast<int>(c.num_kv_heads),
                        static_cast<int>(c.head_dim), c.rope_theta, stream_));
    p.Ok(rocm::RingAppend(b.ck.data, b.cv.data, b.k_ring[il].data,
                          b.v_ring[il].data, first, n, kv, b.ring, stream_));
  }
  if (!p.ok())
    return Fail(error_msg, "drafter injection: " + p.error());
  return true;
}

bool DraftContext::InjectHost(std::span<const float> features,
                              std::uint32_t first, std::string* error_msg) {
  const std::size_t width = drafter_->config().FeatureSize();
  if (features.size() % width != 0)
    return Fail(error_msg, "feature rows must hold FeatureSize floats");
  const int n = static_cast<int>(features.size() / width);
  Buffers& b = *b_;
  if (b.features.count < features.size() &&
      !b.features.Allocate(features.size()))
    return Fail(error_msg, "could not allocate the feature buffer");
  const int pos = static_cast<int>(first);
  if (hipMemcpyAsync(b.features.data, features.data(),
                     features.size() * sizeof(float), hipMemcpyHostToDevice,
                     stream_) != hipSuccess ||
      hipMemcpyAsync(b.first.data, &pos, sizeof(pos), hipMemcpyHostToDevice,
                     stream_) != hipSuccess)
    return Fail(error_msg, "could not upload features");
  if (!Inject(b.features.data, b.first.data, n, error_msg))
    return false;
  if (hipStreamSynchronize(stream_) != hipSuccess)
    return Fail(error_msg, "drafter injection failed");
  return true;
}

bool DraftContext::Draft(std::int32_t anchor, std::uint32_t pos0,
                         DraftLattice* out, std::span<float> logits,
                         std::string* error_msg) {
  const DrafterConfig& c = drafter_->config();
  const rocm::DeviceDrafter& w = drafter_->device();
  const rocm::DeviceModel& trunk = *trunk_;
  Buffers& b = *b_;
  const int T = static_cast<int>(c.block_size);
  const int S = static_cast<int>(c.Slots());
  const int K = static_cast<int>(c.lc_top_k);
  const int N = S * K;
  const int H = static_cast<int>(c.hidden_size);
  const int L = static_cast<int>(c.lc_hidden);
  const int vocab = trunk.output().rows;
  if (pos0 == 0)
    return Fail(error_msg, "drafting needs a context before the anchor");
  if (!logits.empty() && logits.size() != static_cast<std::size_t>(S) * vocab)
    return Fail(error_msg, "draft logits must hold one row per slot");
  const int pos = static_cast<int>(pos0);
  Pass p(stream_, blas_);
  p.Ok(hipMemcpyAsync(b.tokens.data, &anchor, sizeof(anchor),
                      hipMemcpyHostToDevice, stream_));
  p.Ok(hipMemcpyAsync(b.pos0.data, &pos, sizeof(pos), hipMemcpyHostToDevice,
                      stream_));
  p.Ok(rocm::EmbedQ8_0(trunk.token_embd().data, b.tokens.data, b.x.data, T, H,
                       stream_));
  const auto TH = static_cast<std::size_t>(T) * H;
  const int heads = static_cast<int>(c.num_heads);
  const int kv_heads = static_cast<int>(c.num_kv_heads);
  const int hd = static_cast<int>(c.head_dim);
  const int ff = static_cast<int>(c.ff_size);
  const int groups_kernel = static_cast<int>(c.conv_kernel);
  const int group = static_cast<int>(c.conv_group);
  for (std::uint32_t il = 0; p.ok() && il < c.num_layers; ++il) {
    const rocm::DeviceDrafterLayer& l = w.layers[il];
    // Attention with the dynamic convolution around it.
    p.Ok(rocm::RmsNorm(b.x.data, l.attn_norm.f32(), b.h.data, T, H, c.rms_eps,
                       stream_));
    p.Bf16(l.attn_conv_proj, b.h.data, b.dyn.data, T, nullptr);
    p.Ok(rocm::DflashConv(b.h.data, b.dyn.data, l.attn_conv_base.f32(),
                          b.hc.data, T, T, H, group, groups_kernel, 0,
                          stream_));
    p.Bf16(l.attn_q, b.hc.data, b.q.data, T, nullptr);
    p.Bf16(l.attn_k, b.hc.data, b.k.data, T, nullptr);
    p.Bf16(l.attn_v, b.hc.data, b.v.data, T, nullptr);
    p.Ok(rocm::RmsNorm(b.q.data, l.attn_q_norm.f32(), b.q.data, T * heads, hd,
                       c.rms_eps, stream_));
    p.Ok(rocm::RmsNorm(b.k.data, l.attn_k_norm.f32(), b.k.data, T * kv_heads,
                       hd, c.rms_eps, stream_));
    p.Ok(rocm::RopeNeox(b.q.data, b.pos0.data, T, heads, hd, c.rope_theta,
                        stream_));
    p.Ok(rocm::RopeNeox(b.k.data, b.pos0.data, T, kv_heads, hd, c.rope_theta,
                        stream_));
    p.Ok(rocm::RingAppend(b.k.data, b.v.data, b.k_ring[il].data,
                          b.v_ring[il].data, b.pos0.data, T,
                          static_cast<int>(c.KvDim()), b.ring, stream_));
    p.Ok(rocm::DrafterAttention(
        b.q.data, b.k_ring[il].data, b.v_ring[il].data, l.attn_sinks.f32(),
        b.ctx.data, b.pos0.data, T, heads, kv_heads, b.ring,
        static_cast<int>(c.sliding_window),
        1.0F / std::sqrt(static_cast<float>(hd)), stream_));
    p.Bf16(l.attn_out, b.ctx.data, b.o.data, T, nullptr);
    p.Ok(rocm::DflashConv(b.o.data, b.dyn.data, l.attn_conv_base.f32(),
                          b.hc.data, T, T, H, group, groups_kernel, 1,
                          stream_));
    p.Ok(rocm::AddScaled(b.x.data, b.hc.data, 1.0F, nullptr, TH, stream_));
    // SiLU-gated FFN with the dynamic convolution around it.
    p.Ok(rocm::RmsNorm(b.x.data, l.ffn_norm.f32(), b.h.data, T, H, c.rms_eps,
                       stream_));
    p.Bf16(l.ffn_conv_proj, b.h.data, b.dyn.data, T, nullptr);
    p.Ok(rocm::DflashConv(b.h.data, b.dyn.data, l.ffn_conv_base.f32(),
                          b.hc.data, T, T, H, group, groups_kernel, 0,
                          stream_));
    p.Quant(l.ffn_gate, b.hc.data, b.gate.data, T, "ffn_gate");
    p.Quant(l.ffn_up, b.hc.data, b.up.data, T, "ffn_up");
    p.Ok(rocm::SiluGate(b.gate.data, l.ffn_gate.scale, b.up.data,
                        l.ffn_up.scale, b.act.data,
                        static_cast<std::size_t>(T) * ff, stream_));
    p.Quant(l.ffn_down, b.act.data, b.o.data, T, "ffn_down");
    p.Ok(rocm::ScaleBias(b.o.data, l.ffn_down.scale, nullptr, T, H,
                         rocm::Activation::kNone, stream_));
    p.Ok(rocm::DflashConv(b.o.data, b.dyn.data, l.ffn_conv_base.f32(),
                          b.hc.data, T, T, H, group, groups_kernel, 1,
                          stream_));
    p.Ok(rocm::AddScaled(b.x.data, b.hc.data, 1.0F, nullptr, TH, stream_));
  }
  // The drafted slots (block rows 1..) through the trunk's head.
  p.Ok(rocm::RmsNorm(b.x.data, w.output_norm.f32(), b.hidden.data, T, H,
                     c.rms_eps, stream_));
  const float* slots_hidden = b.hidden.data + H;
  p.Quant(trunk.output(), slots_hidden, b.logits.data, S, "output");

  // LiLiCorr: candidates, their features and states.
  p.Ok(rocm::TopK8(b.logits.data, S, vocab, b.cand.data, b.values.data,
                   b.lse.data, stream_));
  p.Ok(rocm::LiliCorrFeatures(b.values.data, b.lse.data, w.feature_norm.f32(),
                              w.feature_norm_b.f32(), b.feat5.data, S,
                              stream_));
  p.F32(w.feature_up, b.feat5.data, b.feat_up.data, N);
  p.Ok(rocm::ScaleBias(b.feat_up.data, 1.0F, w.feature_up_b.f32(), N, L,
                       rocm::Activation::kSilu, stream_));
  p.F32(w.feature_down, b.feat_up.data, b.feat.data, N);
  p.Ok(rocm::ScaleBias(b.feat.data, 1.0F, w.feature_down_b.f32(), N, L,
                       rocm::Activation::kNone, stream_));
  p.Ok(rocm::EmbedQ8_0(trunk.token_embd().data, b.cand.data, b.embd.data, N, H,
                       stream_));
  p.Quant(w.token_proj, b.embd.data, b.token.data, N, "token_proj");
  p.Ok(rocm::ScaleBias(b.token.data, w.token_proj.scale, w.token_proj_b.f32(),
                       N, L, rocm::Activation::kNone, stream_));
  p.Quant(w.pass_proj, slots_hidden, b.pass.data, S, "pass_hidden_proj");
  p.Ok(rocm::ScaleBias(b.pass.data, w.pass_proj.scale, w.pass_proj_b.f32(), S,
                       L, rocm::Activation::kNone, stream_));
  p.Ok(rocm::LiliCorrCombine(b.token.data, b.pass.data, b.feat.data,
                             w.slot_embd.f32(), w.rank_embd.f32(), b.nodes.data,
                             S, L, stream_));
  const float* anchor_row =
      b.anchors.data + static_cast<std::size_t>((pos - 1) % kAnchorRows) * H;
  p.Quant(w.context_proj, anchor_row, b.anchor.data, 1, "context_proj");
  p.Ok(rocm::ScaleBias(b.anchor.data, w.context_proj.scale,
                       w.context_proj_b.f32(), 1, L, rocm::Activation::kNone,
                       stream_));
  const auto NL = static_cast<std::size_t>(N) * L;
  const float node_scale =
      1.0F / std::sqrt(static_cast<float>(L / static_cast<int>(c.lc_heads)));
  for (const rocm::DeviceLiliCorrLayer& l : w.lc_layers) {
    p.Ok(rocm::RmsNorm(b.nodes.data, l.attn_norm.f32(), b.normed.data, N, L,
                       c.rms_eps, stream_));
    p.F32(l.in_proj, b.normed.data, b.qkv.data, N);
    p.Ok(rocm::ScaleBias(b.qkv.data, 1.0F, l.in_proj_b.f32(), N, 3 * L,
                         rocm::Activation::kNone, stream_));
    p.Ok(rocm::NodeAttention(b.qkv.data, w.attention_bias.f32(), b.att.data, N,
                             static_cast<int>(c.lc_heads), node_scale,
                             stream_));
    p.Quant(l.out_proj, b.att.data, b.proj.data, N, "lilicorr attn_output");
    p.Ok(rocm::ScaleBias(b.proj.data, l.out_proj.scale, l.out_proj_b.f32(), N,
                         L, rocm::Activation::kNone, stream_));
    p.Ok(
        rocm::AddScaled(b.nodes.data, b.proj.data, 1.0F, nullptr, NL, stream_));
    p.Ok(rocm::RmsNorm(b.nodes.data, l.mlp_norm.f32(), b.normed.data, N, L,
                       c.rms_eps, stream_));
    p.Quant(l.mlp_up, b.normed.data, b.mlp.data, N, "lilicorr ffn_up");
    p.Ok(rocm::ScaleBias(b.mlp.data, l.mlp_up.scale, l.mlp_up_b.f32(), N,
                         static_cast<int>(c.lc_mlp), rocm::Activation::kSilu,
                         stream_));
    p.Quant(l.mlp_down, b.mlp.data, b.proj.data, N, "lilicorr ffn_down");
    p.Ok(rocm::ScaleBias(b.proj.data, l.mlp_down.scale, l.mlp_down_b.f32(), N,
                         L, rocm::Activation::kNone, stream_));
    p.Ok(
        rocm::AddScaled(b.nodes.data, b.proj.data, 1.0F, nullptr, NL, stream_));
  }
  // Factorized pairwise scores.
  p.Ok(rocm::RmsNorm(b.nodes.data, w.lc_output_norm.f32(), b.normed.data, N, L,
                     c.rms_eps, stream_));
  p.Ok(rocm::RmsNorm(b.anchor.data, w.anchor_norm.f32(), b.anchor.data, 1, L,
                     c.rms_eps, stream_));
  p.Ok(rocm::FactorInput(b.normed.data, b.anchor.data, b.factor_in.data, N, L,
                         stream_));
  p.F32(w.factor_in, b.factor_in.data, b.factors.data, N);
  p.Ok(rocm::ScaleBias(b.factors.data, 1.0F, w.factor_in_b.f32(), N, L,
                       rocm::Activation::kSilu, stream_));
  const int F = static_cast<int>(c.lc_factor);
  p.F32(w.out_head, b.factors.data, b.out_head.data, N);
  p.Ok(rocm::BiasL2Norm(b.out_head.data, w.out_head_b.f32(), N, F,
                        c.lc_vector_eps, stream_));
  p.F32(w.in_head, b.factors.data, b.in_head.data, N);
  p.Ok(rocm::BiasL2Norm(b.in_head.data, w.in_head_b.f32(), N, F,
                        c.lc_vector_eps, stream_));
  p.F32(w.anchor_head, b.anchor.data, b.anchor_head.data, 1);
  p.Ok(rocm::BiasL2Norm(b.anchor_head.data, w.anchor_head_b.f32(), 1, F,
                        c.lc_vector_eps, stream_));
  p.Ok(rocm::LatticeScores(b.in_head.data, b.out_head.data, b.anchor_head.data,
                           b.scores.data, S, F, c.lc_logit_scale, stream_));
  out->slots = static_cast<std::uint32_t>(S);
  out->top_k = static_cast<std::uint32_t>(K);
  out->candidates.resize(N);
  out->scores.resize(b.scores.count);
  p.Ok(hipMemcpyAsync(out->candidates.data(), b.cand.data,
                      N * sizeof(std::int32_t), hipMemcpyDeviceToHost,
                      stream_));
  p.Ok(hipMemcpyAsync(out->scores.data(), b.scores.data,
                      b.scores.count * sizeof(float), hipMemcpyDeviceToHost,
                      stream_));
  if (!logits.empty())
    p.Ok(hipMemcpyAsync(logits.data(), b.logits.data,
                        logits.size() * sizeof(float), hipMemcpyDeviceToHost,
                        stream_));
  p.Ok(hipStreamSynchronize(stream_));
  if (!p.ok())
    return Fail(error_msg, "draft: " + p.error());
  return true;
}

std::size_t DraftContext::StateBytes() const noexcept {
  const DrafterConfig& c = drafter_->config();
  return 2 * c.num_layers * static_cast<std::size_t>(b_->ring) * c.KvDim() *
             sizeof(__half) +
         static_cast<std::size_t>(kAnchorRows) * c.hidden_size * sizeof(float);
}

bool DraftContext::SaveState(std::span<std::uint8_t> out) const {
  if (out.size() != StateBytes())
    return false;
  const Buffers& b = *b_;
  std::uint8_t* dst = out.data();
  bool ok = true;
  auto copy = [&](const void* src, std::size_t bytes) {
    ok = ok && hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost,
                              stream_) == hipSuccess;
    dst += bytes;
  };
  for (std::size_t il = 0; il < b.k_ring.size(); ++il) {
    copy(b.k_ring[il].data, b.k_ring[il].count * sizeof(__half));
    copy(b.v_ring[il].data, b.v_ring[il].count * sizeof(__half));
  }
  copy(b.anchors.data, b.anchors.count * sizeof(float));
  return ok && hipStreamSynchronize(stream_) == hipSuccess;
}

bool DraftContext::RestoreState(std::span<const std::uint8_t> in) {
  if (in.size() != StateBytes())
    return false;
  Buffers& b = *b_;
  const std::uint8_t* src = in.data();
  bool ok = true;
  auto copy = [&](void* dst, std::size_t bytes) {
    ok = ok && hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice,
                              stream_) == hipSuccess;
    src += bytes;
  };
  for (std::size_t il = 0; il < b.k_ring.size(); ++il) {
    copy(b.k_ring[il].data, b.k_ring[il].count * sizeof(__half));
    copy(b.v_ring[il].data, b.v_ring[il].count * sizeof(__half));
  }
  copy(b.anchors.data, b.anchors.count * sizeof(float));
  return ok && hipStreamSynchronize(stream_) == hipSuccess;
}

}  // namespace gufo::models::nemotron35_bolt
