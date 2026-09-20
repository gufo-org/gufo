#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include "qfn_mmq.h"
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

using core::GgmlType;

void AssignError(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

bool Check(hipError_t err, const char* what, std::string* error_msg) {
  if (err == hipSuccess) {
    return true;
  }
  AssignError(error_msg, std::string(what) + ": " + hipGetErrorString(err));
  return false;
}

/// Allocates `count` elements of T and records the allocation.
template<typename T>
T* Alloc(std::vector<void*>& allocations, std::size_t count,
         std::string* error_msg, std::size_t* allocated_bytes = nullptr) {
  void* p = nullptr;
  const std::size_t bytes = std::max<std::size_t>(1, count) * sizeof(T);
  if (hipMalloc(&p, bytes) != hipSuccess) {
    AssignError(error_msg,
                "hipMalloc of " + std::to_string(bytes) + " bytes failed");
    allocations.push_back(nullptr);
    return nullptr;
  }
  (void)hipMemset(p, 0, bytes);
  allocations.push_back(p);
  if (allocated_bytes)
    *allocated_bytes += bytes;
  return static_cast<T*>(p);
}

WeightType SmallType(GgmlType type) {
  switch (type) {
    case GgmlType::kBF16:
      return WeightType::kBF16;
    case GgmlType::kF16:
      return WeightType::kF16;
    case GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    case GgmlType::kF32:
      return WeightType::kF32;
    default:
      throw std::logic_error("unsupported Flash-Next matrix format");
  }
}

// The tier's tiled kernels compute whole column tiles; below this width the
// matrix-vector kernels read each weight once per row and win outright.
constexpr std::uint32_t kVecBatch = 8;
// Keep prompt arithmetic independent of chunk width. A scoped, per-thread
// policy also lets the output head retain its existing logits-row arithmetic.
thread_local bool prefill_phase = false;
struct PrefillPhase {
  bool previous;
  explicit PrefillPhase(bool enabled) : previous(prefill_phase) {
    prefill_phase = enabled;
  }
  ~PrefillPhase() { prefill_phase = previous; }
};
bool MatrixRows(std::uint32_t rows) {
  return prefill_phase || rows > kVecBatch;
}
bool ExpertMatrixRows(std::uint32_t rows) {
  return prefill_phase || rows > 4 * kVecBatch;
}

/// Key-tile splits per row of a narrow attention batch (decode at depth).
constexpr std::uint32_t kAttnSplits = 8;
/// Wide dense Q8_0 projections take the F16 WMMA GEMM (F16 activation
/// rows, weights dequantized as they are staged, no per-block scaling) when
/// its 256-row tiles fill the device and the K sweep is short enough that
/// the F16 rows stay cache-resident: measured per shape at 2048 tokens
/// (W8A8 -> F16 ms): 16384x2560 5.7 -> 4.6, 13312x2560 4.3 -> 3.9,
/// 10240x320 1.02 -> 0.79, 2560x2560 0.81 -> 0.74, 2560x640 0.24 -> 0.21;
/// against 2560x6144 1.8 -> 2.1+, 640x2560 0.19 -> 0.23 and 320x10240
/// 0.40 -> 1.3, which keep the int8 route.
constexpr std::uint32_t kDenseF16MinRows = 2048;
constexpr std::uint32_t kDenseF16MaxCols = 2560;

/// Token rows per routed F16 expert GEMM tile: the narrow tile for batches
/// whose buckets pad to one or two 16-row tiles, the wide one when the
/// mean bucket fills most of it (the weight dequantization is per tile).
constexpr std::uint32_t kRoutedTileRowsNarrow = 16;
constexpr std::uint32_t kRoutedTileRowsWide = 48;

std::uint32_t RoutedTileRows(std::size_t slots, const Config& c) {
  return slots >= static_cast<std::size_t>(16) * c.num_experts
             ? kRoutedTileRowsWide
             : kRoutedTileRowsNarrow;
}

/// Upper bound on launched routed tiles: every 16-padded bucket contributes
/// at most one partial tile beyond its rows.
std::size_t RoutedTileCapacity(std::size_t slots, const Config& c) {
  return (slots + static_cast<std::size_t>(c.num_experts) * 15) /
             kRoutedTileRowsNarrow +
         c.num_experts + 1;
}

std::uint32_t IndexerCapacity(const Config& c, std::uint32_t batch,
                              std::uint32_t context) {
  return static_cast<std::uint32_t>(std::bit_ceil(std::min<std::uint64_t>(
      context, std::uint64_t{c.indexer_top_k} + batch)));
}

}  // namespace

Session::~Session() {
  TrimRollback(0);
  for (auto& [key, exec] : graphs_) {
    (void)hipGraphExecDestroy(exec);
  }
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

void Session::RestoreVisionLayout(const qwen::vision::RopeLayout& layout,
                                  hipStream_t stream) {
  layout.Validate(max_context_);
  const auto* previous = vision_input_.rope();
  vision_input_.RestoreLayout(layout, stream);
  for (auto& attention : attention_)
    attention.rope = vision_input_.rope();
  if (previous != vision_input_.rope()) {
    for (const auto& [key, graph] : graphs_)
      (void)hipGraphExecDestroy(graph);
    graphs_.clear();
    warmed_.clear();
  }
}

void Session::ConfigureVision(
    std::shared_ptr<const qwen::vision::Prompt> prompt,
    std::shared_ptr<qwen::vision::Encoder> encoder, hipStream_t stream) {
  if (prompt)
    prompt->rope.Validate(max_context_);
  const auto* previous = vision_input_.rope();
  vision_input_.Configure(std::move(prompt), std::move(encoder), stream);
  for (auto& attention : attention_)
    attention.rope = vision_input_.rope();
  if (previous != vision_input_.rope()) {
    for (const auto& [key, graph] : graphs_)
      (void)hipGraphExecDestroy(graph);
    graphs_.clear();
    warmed_.clear();
  }
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  cancelled_ = false;
  is_cancelled_ = std::move(check);
  vision_input_.SetCancellationCheck(is_cancelled_);
}

bool Session::CheckCancellation(std::string* error) const {
  try {
    cancelled_ = cancelled_ || (is_cancelled_ && is_cancelled_());
  } catch (...) {
    cancelled_ = true;
  }
  if (cancelled_) {
    AssignError(error, "generation cancelled");
    return false;
  }
  return true;
}

std::size_t Session::AllocatedBytes() const noexcept {
  return allocated_bytes_ + rollback_bytes_ + vision_input_.Bytes();
}

void Session::TrimRollback(std::uint32_t depth) noexcept {
  if (depth >= rollback_depth_)
    return;
  // Keep graphs whose rows still exist. Deeper verifier graphs capture
  // discarded pointers; ordinary decode and MTP graphs do not.
  for (auto it = graphs_.begin(); it != graphs_.end();) {
    if ((it->first & (std::uint64_t{1} << 32)) != 0 &&
        (it->first & (std::uint64_t{1} << 40)) == 0 &&
        (it->first & 0xFFFFU) > depth + 1) {
      (void)hipGraphExecDestroy(it->second);
      it = graphs_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto row = depth; row < rollback_depth_; ++row) {
    (void)hipFree(rollback_allocations_[row]);
    for (auto& layer : linear_) {
      layer.conv_snapshots.rows[row] = nullptr;
      layer.state_snapshots.rows[row] = nullptr;
    }
    ple_snapshots_.rows[row] = nullptr;
  }
  rollback_allocations_.resize(depth);
  rollback_depth_ = depth;
  rollback_bytes_ =
      owner_->SessionBytes(core::SessionMode::kAutoregressive, max_context_,
                           depth) -
      owner_->SessionBytes(core::SessionMode::kAutoregressive, max_context_, 0);
  ngram_snapshots_.resize(depth);
}

void Session::Reset() {
  TrimRollback(0);
  position_ = 0;
  spec_tokens_ = 0;
  ngram_.Reset();
  mtp_.position = 0;
  mtp_.blocks = 0;
  const Config& c = owner_->config();
  for (auto& l : linear_) {
    if (l.state != nullptr) {
      (void)hipMemset(l.conv_state, 0,
                      static_cast<std::size_t>(c.ssm_conv_kernel - 1) *
                          c.SsmConvChannels() * sizeof(float));
      (void)hipMemset(l.state, 0,
                      static_cast<std::size_t>(c.ssm_num_v_heads) *
                          c.ssm_head_dim * c.ssm_head_dim * sizeof(float));
    }
  }
  blocks_ = 0;
  if (ple_history_ != nullptr) {
    (void)hipMemset(ple_history_, 0,
                    static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim() *
                        sizeof(float));
  }
}

Executor::~Executor() {
  // Readers write pinned staging memory; drain them before freeing it,
  // including when a forward failed before reaching PLE.
  if (ple_pending_) {
    (void)ngram_->WaitRead();
  }
  (void)hipFree(batch_logits_);
  (void)hipHostFree(batch_gdn_host_);
  (void)hipHostFree(batch_controls_);
  (void)hipHostFree(batch_candidates_host_);
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
  for (void* p :
       {static_cast<void*>(host_emb_), static_cast<void*>(control_host_),
        static_cast<void*>(tokens_host_), static_cast<void*>(logits_host_),
        static_cast<void*>(mtp_token_host_), static_cast<void*>(counts_host_),
        static_cast<void*>(mtp_candidates_host_),
        static_cast<void*>(tiles_host_)}) {
    if (p != nullptr) {
      (void)hipHostFree(p);
    }
  }
  if (blas_ != nullptr) {
    (void)hipblasDestroy(blas_);
  }
  if (counts_ready_ != nullptr) {
    (void)hipEventDestroy(counts_ready_);
  }
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
  }
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           NgramTable* ngram, Options options,
                                           std::string* error_msg) {
  std::unique_ptr<Executor> e(new Executor());
  e->model_ = &model;
  e->ngram_ = ngram;
  e->options_ = options;
  e->options_.max_batch = std::max<std::uint32_t>(1, options.max_batch);
  e->options_.max_logit_rows = std::clamp<std::uint32_t>(
      options.max_logit_rows, 1, e->options_.max_batch);
  e->options_.max_speculative = std::clamp<std::uint32_t>(
      options.max_speculative, 1, e->options_.max_logit_rows);
  if (qfn_mmq_init(0) != 0) {
    AssignError(error_msg, "quantized GEMM tier initialization failed");
    return nullptr;
  }
  if (!Check(hipStreamCreate(&e->stream_), "hipStreamCreate", error_msg)) {
    return nullptr;
  }
  if (!Check(hipEventCreateWithFlags(&e->counts_ready_, hipEventDisableTiming),
             "expert counts event", error_msg)) {
    return nullptr;
  }
  if (hipblasCreate(&e->blas_) != HIPBLAS_STATUS_SUCCESS ||
      hipblasSetStream(e->blas_, e->stream_) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error_msg, "hipblasCreate failed");
    return nullptr;
  }
  const Config& c = model.config();
  const std::size_t T = e->options_.max_batch;
  e->blaslt_ = BlasLt::Create(e->stream_, error_msg);
  if (e->blaslt_ == nullptr) {
    return nullptr;
  }
  const std::size_t hc_dim = c.HcDim();
  const std::size_t hidden = c.hidden_size;
  const std::size_t slots = T * c.num_experts_used;
  auto& a = e->allocations_;
  Scratch& s = e->s_;
  auto f32 = [&](std::size_t n) { return Alloc<float>(a, n, error_msg); };
  s.tokens = Alloc<std::int32_t>(a, T, error_msg);
  s.x_half = Alloc<std::uint16_t>(a, T * model.max_half_cols(), error_msg);
  for (void*& slot : s.x_q8) {
    slot = Alloc<std::uint8_t>(
        a,
        qfn_mmq_q8_1_bytes(static_cast<int>(kVecBatch),
                           static_cast<int>(model.max_q8_cols())),
        error_msg);
  }
  s.x_q8t =
      Alloc<std::uint8_t>(a, Q8TiledBytes(T, model.max_q8_cols()), error_msg);
  s.res = f32(T * hc_dim);
  s.xn = f32(T * hc_dim);
  s.xn_half = Alloc<__half>(a, T * hc_dim, error_msg);
  s.xn_q8t = Alloc<std::uint8_t>(a, Q8TiledBytes(T, hc_dim), error_msg);
  s.lo = f32(T * c.hc_low_rank);
  s.hc_gate = f32(T * hc_dim);
  s.mixed = f32(T * hidden);
  s.inject = f32(T * c.hc_count * HcInjectParts(hidden));
  s.block_out = f32(T * hidden);
  // Stacked and separate SSM projections are mutually exclusive. MTP's
  // projected embedding is consumed before attention and reuses this space.
  s.qkvz = f32(T * std::max<std::size_t>({c.SsmConvChannels() + c.SsmValueDim(),
                                          c.ple_layer >= 0 ? hc_dim : 0}));
  s.qkv = s.qkvz;
  s.z = s.qkvz != nullptr ? s.qkvz + T * c.SsmConvChannels() : nullptr;
  s.alpha_beta = f32(T * 2 * c.ssm_num_v_heads);
  s.conv_scratch = f32((T + c.ssm_conv_kernel) * c.SsmConvChannels());
  s.qn = f32(T * c.SsmKeyDim());
  s.kn = f32(T * c.SsmKeyDim());
  s.gdn_raw = f32(T * c.SsmValueDim());
  s.gdn_out = f32(T * c.SsmValueDim());
  s.qg = f32(T * (2 * c.AttentionQDim() + 2 * c.AttentionKvDim()));
  s.q = f32(T * c.AttentionQDim());
  s.attn_gate = f32(T * c.AttentionQDim());
  s.k = f32(T * c.AttentionKvDim());
  s.v = f32(T * c.AttentionKvDim());
  s.iq = f32(T * c.indexer_heads * c.indexer_head_dim);
  s.ik = f32(T * c.indexer_head_dim);
  const std::uint32_t max_blocks =
      (c.context_length + c.compress_ratio - 1) / c.compress_ratio;
  e->mask_words_ = (max_blocks + 31) / 32;
  s.mask = Alloc<std::uint32_t>(a, T * e->mask_words_, error_msg);
  s.scores =
      f32(static_cast<std::size_t>(e->select_chunk_) * e->mask_words_ * 32);
  s.ctx = f32(T * c.AttentionQDim());
  s.attn_partials = f32(static_cast<std::size_t>(kVecBatch) * c.num_heads *
                        kAttnSplits * (c.head_dim + 2));
  if (c.ple_layer >= 0) {
    s.ple_emb = f32(T * c.PleEmbeddingDim());
    // PLE finishes before this layer's mixer/SSM. Its gate consumes key
    // and query before normalization/convolution reuse those two buffers.
    // The gated input stays separate until PleInject consumes both outputs.
    s.ple_key = s.hc_gate;
    s.ple_value = s.block_out;
    s.ple_query = s.xn;
    s.ple_gated = s.qkvz;
    s.ple_norm = s.hc_gate;
    s.ple_conv = s.xn;
    s.ple_history_scratch =
        f32(static_cast<std::size_t>(c.PleConvHistory()) * hc_dim);
    void* pinned = nullptr;
    if (!Check(hipHostMalloc(&pinned, T * c.PleEmbeddingDim() * sizeof(float)),
               "pinned n-gram buffer", error_msg)) {
      return nullptr;
    }
    e->host_emb_ = static_cast<float*>(pinned);
    e->host_rows_.resize(T * c.ple_heads);
  }
  s.router = f32(T * (c.num_experts + 1));
  s.ids = Alloc<std::int32_t>(a, slots, error_msg);
  s.expert_counts = Alloc<std::uint32_t>(a, c.num_experts, error_msg);
  {
    const std::size_t compact = RoutedCompactRows(slots, c.num_experts);
    s.routed_bounds = Alloc<std::int32_t>(a, c.num_experts + 1, error_msg);
    s.routed_cursors = Alloc<std::int32_t>(a, c.num_experts, error_msg);
    s.rows_token = Alloc<std::int32_t>(a, compact, error_msg);
    s.rows_slot = Alloc<std::int32_t>(a, compact, error_msg);
    s.routed_tiles =
        Alloc<std::int32_t>(a, 3 * RoutedTileCapacity(slots, c), error_msg);
  }
  s.weights = f32(slots);
  s.gate_e = f32(slots * c.expert_ff);
  s.up_e = f32(slots * c.expert_ff);
  s.down_e = f32(slots * hidden);
  s.shexp_gate = f32(T * c.shared_expert_ff);
  s.shexp_up = f32(T * c.shared_expert_ff);
  s.shexp_out = f32(T * hidden);
  s.shexp_half = Alloc<__half>(a, T * c.shared_expert_ff, error_msg);
  s.logits =
      f32(static_cast<std::size_t>(e->options_.max_logit_rows) * c.vocab_size);
  {
    void* control = nullptr;
    void* tokens = nullptr;
    void* logits = nullptr;
    if (!Check(hipHostMalloc(&control, sizeof(Session::Control)),
               "pinned control buffer", error_msg) ||
        !Check(hipHostMalloc(&tokens, T * sizeof(std::int32_t)),
               "pinned token buffer", error_msg) ||
        !Check(hipHostMalloc(&logits, static_cast<std::size_t>(
                                          e->options_.max_logit_rows) *
                                          c.vocab_size * sizeof(float)),
               "pinned logits buffer", error_msg)) {
      return nullptr;
    }
    void* counts = nullptr;
    if (!Check(hipHostMalloc(&counts, c.num_experts * sizeof(std::uint32_t)),
               "pinned expert counts", error_msg)) {
      return nullptr;
    }
    e->counts_host_ = static_cast<std::uint32_t*>(counts);
    void* tiles = nullptr;
    if (!Check(hipHostMalloc(&tiles, 3 * RoutedTileCapacity(slots, c) *
                                         sizeof(std::int32_t)),
               "pinned routed tile map", error_msg)) {
      return nullptr;
    }
    e->tiles_host_ = static_cast<std::int32_t*>(tiles);
    e->control_host_ = static_cast<Session::Control*>(control);
    e->tokens_host_ = static_cast<std::int32_t*>(tokens);
    e->logits_host_ = static_cast<float*>(logits);
  }
  // The wide mixer route (F16 norm for the epilogue, tiled Q8 norm for the
  // W8A8 down projection) needs the four-stream geometry, 32-wide blocks and
  // a Q8_0 down projection.
  e->wide_mixer_ = c.hc_count == 4 && c.hidden_size % 32 == 0 &&
                   !model.layers().empty() &&
                   model.layers()[0].hc_ffn.down.type == GgmlType::kQ8_0;
  if (model.has_mtp()) {
    // The trunk's kept rows are session-owned. Its transient residual and
    // mixer buffers are free while MTP constructs its input. The split
    // projections consume h/embd before the first mixer rewrites xn/mixed.
    s.mtp_h = s.xn;
    s.mtp_embd = s.mixed;
    s.mtp_eproj = s.qkvz;
    s.mtp_res = s.res;
    s.mtp_argmax = Alloc<ArgmaxCandidate>(a, kArgmaxParts, error_msg);
    s.mtp_token = Alloc<std::int32_t>(a, 1, error_msg);
    const auto candidate_ids = MtpCandidateWorkspaceSize(c.vocab_size);
    s.mtp_ids = Alloc<std::uint32_t>(a, candidate_ids, error_msg);
    s.mtp_scratch_ids = Alloc<std::uint32_t>(a, candidate_ids, error_msg);
    // Final selection consumes intermediate IDs before writing its scores.
    // Pack those scores behind the 64 returned IDs for one host transfer.
    s.mtp_scores = reinterpret_cast<float*>(s.mtp_ids + kMtpCandidates);
    if (!Check(hipHostMalloc(&e->mtp_token_host_, sizeof(std::int32_t)),
               "pinned draft token", error_msg) ||
        !Check(
            hipHostMalloc(&e->mtp_candidates_host_, sizeof(MtpCandidateLogits)),
            "pinned draft candidates", error_msg)) {
      return nullptr;
    }
    std::construct_at(e->mtp_candidates_host_);
    e->mtp_candidates_host_->size =
        std::min<std::size_t>(c.vocab_size, kMtpCandidates);
  }
  for (void* p : a) {
    if (p == nullptr) {
      return nullptr;
    }
  }
  return e;
}

std::unique_ptr<Session> Executor::CreateSession(core::SessionMode mode,
                                                 std::uint32_t max_context,
                                                 std::string* error_msg) const {
  std::unique_ptr<Session> s(new Session());
  s->owner_ = this;
  s->mtp_enabled_ = mode == core::SessionMode::kSpeculative;
  if (s->mtp_enabled_ && !has_mtp()) {
    AssignError(error_msg, "MTP session requires a loaded predictor");
    return nullptr;
  }
  const Config& c = config();
  if (max_context == 0 || max_context > c.context_length) {
    AssignError(error_msg, "session context exceeds the model context");
    return nullptr;
  }
  s->max_context_ = max_context;
  // Before sparse attention starts, pooling can lag by indexer_top_k
  // rows. Afterwards only an incomplete block precedes the current batch.
  // Completed block keys remain in block_k; their raw rows are dead.
  s->index_capacity_ = IndexerCapacity(c, options_.max_batch, max_context);
  s->linear_.resize(c.num_layers);
  s->attention_.resize(c.num_layers);
  auto& a = s->allocations_;
  const std::size_t kv_row = c.AttentionKvDim();
  const std::size_t conv_elems =
      static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  const std::size_t state_elems = static_cast<std::size_t>(c.ssm_num_v_heads) *
                                  c.ssm_head_dim * c.ssm_head_dim;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (c.IsLinearLayer(il)) {
      auto& l = s->linear_[il];
      l.conv_state =
          Alloc<float>(a, conv_elems, error_msg, &s->allocated_bytes_);
      l.state = Alloc<float>(a, state_elems, error_msg, &s->allocated_bytes_);
    } else {
      auto& at = s->attention_[il];
      at.k_cache =
          Alloc<__half>(a, static_cast<std::size_t>(max_context) * kv_row,
                        error_msg, &s->allocated_bytes_);
      at.v_cache =
          Alloc<__half>(a, static_cast<std::size_t>(max_context) * kv_row,
                        error_msg, &s->allocated_bytes_);
      at.index_k = Alloc<float>(
          a, static_cast<std::size_t>(s->index_capacity_) * c.indexer_head_dim,
          error_msg, &s->allocated_bytes_);
      at.block_k = Alloc<__half>(
          a,
          static_cast<std::size_t>(max_context / c.compress_ratio + 1) *
              c.indexer_head_dim,
          error_msg, &s->allocated_bytes_);
    }
  }
  if (c.ple_layer >= 0) {
    const std::size_t hist =
        static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim();
    s->ple_history_ = Alloc<float>(a, hist, error_msg, &s->allocated_bytes_);
  }
  s->control_ = Alloc<Session::Control>(a, 1, error_msg, &s->allocated_bytes_);
  if (s->mtp_enabled_) {
    s->mtp_.target_hidden = Alloc<float>(
        a, static_cast<std::size_t>(options_.max_speculative) * c.HcDim(),
        error_msg, &s->allocated_bytes_);
    s->mtp_.k_cache =
        Alloc<__half>(a, static_cast<std::size_t>(max_context) * kv_row,
                      error_msg, &s->allocated_bytes_);
    s->mtp_.v_cache =
        Alloc<__half>(a, static_cast<std::size_t>(max_context) * kv_row,
                      error_msg, &s->allocated_bytes_);
    s->mtp_.index_k =
        Alloc<float>(a, std::size_t{s->index_capacity_} * c.indexer_head_dim,
                     error_msg, &s->allocated_bytes_);
    s->mtp_.block_k = Alloc<__half>(
        a, std::size_t{max_context / c.compress_ratio + 1} * c.indexer_head_dim,
        error_msg, &s->allocated_bytes_);
    s->mtp_.h = Alloc<float>(a, c.HcDim(), error_msg, &s->allocated_bytes_);
  }
  for (void* p : a) {
    if (p == nullptr) {
      return nullptr;
    }
  }
  // The zero fills above run on the null stream; nothing may read the new
  // buffers until they have landed.
  if (!Check(hipDeviceSynchronize(), "session init", error_msg)) {
    return nullptr;
  }
  return s;
}

bool Executor::EnsureRollback(Session& session, std::uint32_t depth,
                              std::string* error_msg) const {
  if (depth <= session.rollback_depth_)
    return true;
  if (depth >= options_.max_speculative ||
      depth > std::size(session.ple_snapshots_.rows) ||
      session.spec_tokens_ != 0) {
    AssignError(error_msg, "invalid rollback allocation depth");
    return false;
  }
  const auto& c = config();
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t linear =
      c.num_layers - c.num_layers / c.full_attention_interval;
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  session.rollback_allocations_.reserve(depth);
  session.ngram_snapshots_.resize(depth);
  for (auto row = session.rollback_depth_; row < depth; ++row) {
    const auto state = GdnRollbackRowFloats(row, c.ssm_num_k_heads,
                                            c.ssm_num_v_heads, c.ssm_head_dim);
    const auto bytes = (linear * (conv + state) + ple) * sizeof(float);
    float* allocation = nullptr;
    if (!Check(hipMalloc(&allocation, bytes), "rollback allocation", error_msg))
      return false;
    session.rollback_allocations_.push_back(allocation);
    session.rollback_bytes_ += bytes;
    auto* next = allocation;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (!c.IsLinearLayer(il))
        continue;
      session.linear_[il].conv_snapshots.rows[row] = next;
      next += conv;
      session.linear_[il].state_snapshots.rows[row] = next;
      next += state;
    }
    if (ple != 0) {
      session.ple_snapshots_.rows[row] = next;
    }
    session.rollback_depth_ = row + 1;
  }
  return true;
}

std::size_t Executor::SessionBytes(
    core::SessionMode mode, std::uint32_t max_context,
    std::uint32_t rollback_depth) const noexcept {
  const auto& c = config();
  const std::size_t attention = c.num_layers / c.full_attention_interval;
  const std::size_t linear = c.num_layers - attention;
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t state =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  const std::size_t kv =
      2 * std::size_t{max_context} * c.AttentionKvDim() * sizeof(__half);
  const std::size_t index =
      std::size_t{IndexerCapacity(c, options_.max_batch, max_context)} *
          c.indexer_head_dim * sizeof(float) +
      std::size_t{max_context / c.compress_ratio + 1} * c.indexer_head_dim *
          sizeof(__half);
  const auto rollback_state =
      rollback_depth == 0
          ? 0
          : state + (rollback_depth - 1) *
                        GdnRollbackRowFloats(1, c.ssm_num_k_heads,
                                             c.ssm_num_v_heads, c.ssm_head_dim);
  return ((linear * conv + ple) * (rollback_depth + 1) +
          linear * (state + rollback_state)) *
             sizeof(float) +
         attention * (kv + index) + sizeof(Session::Control) +
         (mode == core::SessionMode::kSpeculative
              ? kv + index +
                    std::size_t{options_.max_speculative + 1} * c.HcDim() *
                        sizeof(float)
              : 0);
}

std::size_t Executor::DeferredScratchBytes() const {
  const auto& c = config();
  std::size_t bytes = batch_logits_ == nullptr
                          ? std::size_t{8} *
                                std::min(8U, options_.max_logit_rows) *
                                c.vocab_size * sizeof(float)
                          : 0;
  return bytes;
}

/// Column-tile width for the routed expert GEMMs: the tile at or above twice
/// the mean bucket, so most experts fit one tile with little padding.
int RoutedTileCols(std::uint32_t n_tokens, std::uint32_t n_used,
                   std::uint32_t n_experts) {
  const std::uint32_t mean =
      std::max<std::uint32_t>(1, n_tokens * n_used / std::max(n_experts, 1u));
  for (int cols = 16; cols < 80; cols += 16) {
    if (static_cast<std::uint32_t>(cols) >= 2 * mean) {
      return cols;
    }
  }
  return 80;
}

bool Executor::Quantize(const float* x, std::uint32_t n_tokens, std::uint32_t k,
                        Q8Input* q, std::string* error_msg) const {
  q->x = x;
  q->data = nullptr;
  q->n = n_tokens;
  q->k = k;
  if (MatrixRows(n_tokens)) {
    return true;  // the tiled path quantizes per call
  }
  // Two slots alternate, so an input stays valid across one other
  // quantization; captured graphs replay the same alternation.
  void* slot = s_.x_q8[q8_slot_];
  q8_slot_ ^= 1u;
  if (qfn_mmq_quantize_q8_1(x, slot, static_cast<int>(n_tokens),
                            static_cast<int>(k), stream_) != 0) {
    AssignError(error_msg, "activation quantization failed");
    return false;
  }
  q->data = slot;
  return true;
}

bool Executor::Dense(const DeviceTensor& w, const Q8Input& q, float* out,
                     std::string* error_msg) const {
  if (w.type == GgmlType::kQ8_0 && q.data != nullptr) {
    if (w.cols != q.k) {
      AssignError(error_msg, "quantized input width mismatch");
      return false;
    }
    if (qfn_mmq_q8_0_dense_vec_preq(
            w.data, nullptr, q.data, out, static_cast<int>(w.rows),
            static_cast<int>(q.n), static_cast<int>(w.cols), stream_) != 0) {
      AssignError(error_msg, "Q8_0 GEMV failed");
      return false;
    }
    return true;
  }
  return Dense(w, q.x, out, q.n, error_msg);
}

bool Executor::GatedDense(const DeviceTensor& up, const DeviceTensor& gate,
                          const float* x, float* out, std::uint32_t n_tokens,
                          const DeviceTensor* down,
                          std::string* error_msg) const {
  // Decode and verification use the same fused projections and activation.
  // A different SwiGLU rounding can change later activation quantization.
  if (!MatrixRows(n_tokens) && up.type == GgmlType::kQ8_0 &&
      gate.type == GgmlType::kQ8_0 && up.rows == gate.rows &&
      up.cols == gate.cols) {
    Q8Input xq;
    if (!Quantize(x, n_tokens, up.cols, &xq, error_msg)) {
      return false;
    }
    if (qfn_mmq_q8_0_dense_vec_preq(up.data, gate.data, xq.data, out,
                                    static_cast<int>(up.rows),
                                    static_cast<int>(n_tokens),
                                    static_cast<int>(up.cols), stream_) != 0) {
      AssignError(error_msg, "gated Q8_0 GEMV failed");
      return false;
    }
    return true;
  }
  // Swiglu is in place over its first operand.
  if (!Dense(gate, x, out, n_tokens, error_msg) ||
      !Dense(up, x, s_.shexp_gate, n_tokens, error_msg)) {
    return false;
  }
  // A wide batch writes the down projection's staged input directly (the
  // F32 rows are read by nothing else): F16 rows for the F16 route, else
  // the tiled Q8 layout; the cache lets Dense skip its activation pass.
  if (down != nullptr && down->cols == up.rows && MatrixRows(n_tokens) &&
      n_tokens <= options_.max_batch) {
    if (DenseF16Route(*down, n_tokens)) {
      // A private F16 buffer: s_.x_half keeps the narrowed token rows the
      // router left there, which the routed experts read next.
      SwigluHalf(out, s_.shexp_gate, s_.shexp_half,
                 static_cast<std::size_t>(n_tokens) * up.rows, stream_);
      shexp_half_ready_ = true;
      return true;
    }
    if (down->type == GgmlType::kQ8_0 &&
        SwigluQ8Tiled(out, s_.shexp_gate, s_.x_q8t, n_tokens, up.rows,
                      stream_)) {
      q8t_src_ = out;
      q8t_rows_ = n_tokens;
      q8t_cols_ = up.rows;
      return true;
    }
  }
  Swiglu(out, s_.shexp_gate, static_cast<std::size_t>(n_tokens) * up.rows,
         stream_);
  return true;
}

void Executor::PrepareHalfInput(const float* x, std::uint32_t rows,
                                std::uint32_t cols) const {
  if (half_src_ == x && half_rows_ == rows && half_cols_ == cols &&
      !half_bf16_) {
    return;
  }
  NarrowActivations(x, s_.x_half, false, static_cast<std::size_t>(rows) * cols,
                    stream_);
  half_src_ = x;
  half_rows_ = rows;
  half_cols_ = cols;
  half_bf16_ = false;
}

bool Executor::DenseF16Route(const DeviceTensor& w,
                             std::uint32_t n_tokens) const {
  return w.type == GgmlType::kQ8_0 && MatrixRows(n_tokens) &&
         w.rows >= kDenseF16MinRows && w.cols <= kDenseF16MaxCols &&
         w.cols <= model_->max_half_cols();
}

bool Executor::Dense(const DeviceTensor& w, const float* x, float* out,
                     std::uint32_t n_tokens, std::string* error_msg) const {
  if (w.type == GgmlType::kQ8_0) {
    if (!MatrixRows(n_tokens)) {
      Q8Input q;
      return Quantize(x, n_tokens, w.cols, &q, error_msg) &&
             Dense(w, q, out, error_msg);
    }
    // Wide batches: the F16 route for the shapes it wins (F16 rows in
    // s_.x_half, often left there by a producer), else activations
    // quantized per 32-wide block into the tiled layout and the int8 WMMA
    // GEMM. Both staging buffers hold max_batch rows; a wider call (the
    // draft block folds its streams into rows) runs in pieces.
    const bool f16 = DenseF16Route(w, n_tokens);
    const std::uint32_t piece = static_cast<std::uint32_t>(options_.max_batch);
    for (std::uint32_t r0 = 0; r0 < n_tokens; r0 += piece) {
      const std::uint32_t rows = std::min(piece, n_tokens - r0);
      // The staging buffer may already hold this input (a producer wrote
      // it, or the previous projection read the same rows).
      const float* src = x + static_cast<std::size_t>(r0) * w.cols;
      if (f16) {
        PrepareHalfInput(src, rows, w.cols);
        if (!DenseF16Gemm(w.data, static_cast<const __half*>(s_.x_half),
                          out + static_cast<std::size_t>(r0) * w.rows, rows,
                          w.rows, w.cols, stream_)) {
          AssignError(error_msg, "dense F16 GEMM failed");
          return false;
        }
        continue;
      }
      if (!(q8t_src_ == src && q8t_rows_ == rows && q8t_cols_ == w.cols)) {
        QuantizeQ8Tiled(src, s_.x_q8t, rows, w.cols, stream_);
        q8t_src_ = src;
        q8t_rows_ = rows;
        q8t_cols_ = w.cols;
      }
      if (!W8A8Gemm(w.data, s_.x_q8t,
                    out + static_cast<std::size_t>(r0) * w.rows, rows, w.rows,
                    w.cols, stream_)) {
        AssignError(error_msg, "W8A8 GEMM failed");
        return false;
      }
    }
    return true;
  }
  if (!MatrixRows(n_tokens)) {
    SmallGemm(w.data, SmallType(w.type), x, out, n_tokens, w.rows, w.cols,
              stream_);
    return true;
  }
  // Wide batches of the unquantized projections (router, alpha/beta,
  // indexer): out[t][m] = sum_k w[m][k] x[t][k].
  const int m = static_cast<int>(w.rows);
  const int k = static_cast<int>(w.cols);
  const int n = static_cast<int>(n_tokens);
  if (w.type == GgmlType::kF32) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    if (hipblasSgemm(blas_, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha,
                     static_cast<const float*>(w.data), k, x, k, &beta, out,
                     m) != HIPBLAS_STATUS_SUCCESS) {
      AssignError(error_msg, "hipBLAS GEMM failed");
      return false;
    }
    return true;
  }
  const bool bf16 = w.type == GgmlType::kBF16;
  const hipDataType type = bf16 ? HIP_R_16BF : HIP_R_16F;
  if (!(half_src_ == x && half_rows_ == n_tokens && half_cols_ == w.cols &&
        half_bf16_ == bf16)) {
    NarrowActivations(x, s_.x_half, bf16, static_cast<std::size_t>(n) * k,
                      stream_);
    half_src_ = x;
    half_rows_ = n_tokens;
    half_cols_ = w.cols;
    half_bf16_ = bf16;
  }
  return blaslt_->Gemm(w.data, s_.x_half, out, type, m, n, k, error_msg);
}

void Executor::RoutedHints(const DeviceTensor& w,
                           std::uint32_t n_tokens) const {
  // The column grid is bounded by the largest expert bucket and the tile
  // width fitted to the whole distribution (see RouteHints); the fallback
  // bound is the token count with a tile near twice the mean bucket.
  if (routed_max_rows_ > 0) {
    qfn_mmq_set_routed_max_expert_rows(static_cast<int>(routed_max_rows_));
    qfn_mmq_set_routed_tile_cols(routed_tile_cols_);
    return;
  }
  qfn_mmq_set_routed_max_expert_rows(static_cast<int>(n_tokens));
  qfn_mmq_set_routed_tile_cols(
      RoutedTileCols(n_tokens, config().num_experts_used, w.experts));
}

bool Executor::RouteHints(std::uint32_t n_tokens,
                          std::string* error_msg) const {
  // Every column tile past an expert's bucket still costs a dispatch and a
  // full shared-memory reservation, so the grid is cut to the real largest
  // bucket. Counts are downloaded before the shared expert; wait only for
  // that download while the shared expert continues on the same stream.
  const Config& c = config();
  routed_max_rows_ = 0;
  routed_64_tiles_ = 0;
  routed_pair_tiles_ = 0;
  routed_pair_offset_ = 0;
  routed_pair_rows_ = 64;
  if (!ExpertMatrixRows(n_tokens)) {
    return true;
  }
  if (!Check(hipEventSynchronize(counts_ready_), "expert counts", error_msg)) {
    return false;
  }
  // The F16 expert GEMM launches one block per (expert, row tile of its
  // 16-padded bucket): the map is built here and uploaded ahead of the
  // launches on the same stream.
  std::uint32_t max_rows = 0;
  std::uint32_t n_tiles = 0;
  routed_tile_rows_ = RoutedTileRows(
      static_cast<std::size_t>(n_tokens) * c.num_experts_used, c);
  for (std::uint32_t e = 0; e < c.num_experts; ++e) {
    const std::uint32_t padded = (counts_host_[e] + 15u) / 16u * 16u;
    max_rows = std::max(max_rows, counts_host_[e]);
    for (std::uint32_t j = 0;
         j < (padded + routed_tile_rows_ - 1) / routed_tile_rows_; ++j) {
      tiles_host_[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
  }
  routed_max_rows_ = std::max<std::uint32_t>(1, max_rows);
  routed_n_tiles_ = n_tiles;
  // Append the 64-token map for gate/up and eligible down projections.
  // Wide expert buckets also get a 128-token gate/up map: it amortizes
  // weight decoding, while short buckets retain the cheaper 64-token tile.
  if (n_tokens >= 1024 && routed_tile_rows_ == kRoutedTileRowsWide) {
    for (std::uint32_t e = 0; e < c.num_experts; ++e) {
      const std::uint32_t padded = (counts_host_[e] + 15u) / 16u * 16u;
      for (std::uint32_t j = 0; j < (padded + 63u) / 64u; ++j)
        tiles_host_[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
    routed_64_tiles_ = n_tiles - routed_n_tiles_;
    routed_pair_offset_ = routed_n_tiles_;
    routed_pair_tiles_ = routed_64_tiles_;
    std::uint32_t tiles_128 = 0;
    for (std::uint32_t e = 0; e < c.num_experts; ++e) {
      tiles_128 += (counts_host_[e] + 127u) / 128u;
    }
    if (tiles_128 * 4 <= routed_64_tiles_ * 3) {
      routed_pair_rows_ = 128;
      routed_pair_offset_ = n_tiles;
      routed_pair_tiles_ = tiles_128;
      for (std::uint32_t e = 0; e < c.num_experts; ++e) {
        for (std::uint32_t j = 0; j < (counts_host_[e] + 127u) / 128u; ++j) {
          tiles_host_[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
        }
      }
    }
  }
  routed_tile_cols_ = qfn_mmq_routed_tile_cols_for_counts(
      counts_host_, static_cast<int>(c.num_experts));
  return n_tiles == 0 || Check(hipMemcpyAsync(s_.routed_tiles, tiles_host_,
                                              n_tiles * sizeof(std::int32_t),
                                              hipMemcpyHostToDevice, stream_),
                               "routed tile map upload", error_msg);
}

bool Executor::Experts(const DeviceTensor& w, const float* x,
                       const std::int32_t* ids, float* out,
                       std::uint32_t n_rows, std::uint32_t n_used,
                       std::uint32_t n_tokens, std::string* error_msg) const {
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int E = static_cast<int>(w.experts);
  const int T = static_cast<int>(n_rows);
  const int U = static_cast<int>(n_used);
  // The vector entries loop over column chunks, so the decode-time down
  // projection (top-k rows, one expert each) stays on them as well.
  // Verification keeps the same quantization and reduction as single-token
  // decoding even when top-k expansion produces more than 32 slot rows.
  const bool tiled =
      (prefill_phase || n_rows > 4 * kVecBatch) && MatrixRows(n_tokens);
  if (tiled) {
    RoutedHints(w, n_tokens);
  }
  int rc = -1;
  if (!tiled) {
    rc = qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, x, ids, out, M, K, T,
                         E, U, stream_);
  } else {
    switch (w.type) {
      case GgmlType::kQ4_K:
        rc = qfn_mmq_q4_K_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      case GgmlType::kQ5_K:
        rc = qfn_mmq_q5_K_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      case GgmlType::kQ5_1:
        rc = qfn_mmq_q5_1_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      case GgmlType::kQ8_0:
        rc = qfn_mmq_q8_0_moe_raw(w.data, x, ids, out, M, K, T, E, U, stream_);
        break;
      default:
        break;
    }
  }
  if (rc != 0) {
    AssignError(error_msg, "expert GEMM failed");
    return false;
  }
  return true;
}

bool Executor::GatedExperts(const DeviceTensor& a, const DeviceTensor& b,
                            const float* x, const std::int32_t* ids, float* out,
                            std::uint32_t n_tokens, std::uint32_t n_used,
                            std::string* error_msg) const {
  const bool same_shape = a.type == b.type && a.rows == b.rows &&
                          a.cols == b.cols && a.experts == b.experts;
  if (!MatrixRows(n_tokens) && n_used <= 32 && same_shape &&
      (a.type == GgmlType::kQ4_K || a.type == GgmlType::kQ5_K ||
       a.type == GgmlType::kQ8_0)) {
    if (qfn_mmq_moe_gated_vec(
            static_cast<int>(a.type), a.data, b.data, x, ids, out,
            static_cast<int>(a.rows), static_cast<int>(a.cols),
            static_cast<int>(n_tokens), static_cast<int>(a.experts),
            static_cast<int>(n_used), stream_) != 0) {
      AssignError(error_msg, "gated expert vector projection failed");
      return false;
    }
    return true;
  }
  if (!ExpertMatrixRows(n_tokens) && same_shape) {
    if (qfn_mmq_moe_vec(static_cast<int>(a.type), a.data, x, ids, out,
                        static_cast<int>(a.rows), static_cast<int>(a.cols),
                        static_cast<int>(n_tokens), static_cast<int>(a.experts),
                        static_cast<int>(n_used), stream_, b.data,
                        s_.up_e) != 0) {
      AssignError(error_msg, "expert vector pair GEMM failed");
      return false;
    }
  } else if (same_shape && a.type == GgmlType::kQ4_K) {
    // The wide Q4_K path shares its gather and tiled quantization as well.
    RoutedHints(a, n_tokens);
    if (qfn_mmq_q4_K_moe_pair_unique(
            a.data, b.data, x, ids, out, s_.up_e, static_cast<int>(a.rows),
            static_cast<int>(a.cols), static_cast<int>(n_tokens),
            static_cast<int>(a.experts), static_cast<int>(n_used),
            stream_) != 0) {
      AssignError(error_msg, "expert pair GEMM failed");
      return false;
    }
  } else if (!Experts(a, x, ids, out, n_tokens, n_used, n_tokens, error_msg) ||
             !Experts(b, x, ids, s_.up_e, n_tokens, n_used, n_tokens,
                      error_msg)) {
    return false;
  }
  Swiglu(out, s_.up_e, static_cast<std::size_t>(n_tokens) * n_used * a.rows,
         stream_);
  return true;
}

void Executor::Combine(float* res, const float* gamma,
                       std::uint32_t n_tokens) const {
  const Config& c = config();
  // Wide batches hand the next mixer an F16 norm for its epilogue and the
  // same norm quantized into the tiled Q8 layout for its W8A8 down
  // projection: half the bytes for the combine and the epilogue, and no
  // separate activation pass for the projection.
  xn_half_ = wide_mixer_ && MatrixRows(n_tokens) && gamma != nullptr;
  if (moe_pending_) {
    // The MoE epilogue was deferred to this combine (see Moe).
    moe_pending_ = false;
    const auto* down = reinterpret_cast<const __half*>(s_.down_e);
    if (xn_half_ &&
        HcCombineMoeF16(res, down, s_.weights, s_.shexp_out,
                        s_.router + c.num_experts, c.num_experts + 1,
                        c.num_experts_used, s_.inject, inject_parts_, gamma,
                        s_.xn_half, s_.xn_q8t, n_tokens, c.hidden_size,
                        c.hc_count, c.rms_eps, stream_)) {
      return;
    }
    MoeEpilogueVec4F16(down, s_.weights, s_.shexp_out,
                       s_.router + c.num_experts, c.num_experts + 1,
                       s_.block_out, n_tokens, c.num_experts_used,
                       c.hidden_size, stream_);
  }
  if (xn_half_) {
    HcCombineF16(res, s_.block_out, s_.inject, inject_parts_, gamma, s_.xn_half,
                 s_.xn_q8t, n_tokens, c.hidden_size, c.hc_count, c.rms_eps,
                 stream_);
    return;
  }
  HcCombine(res, s_.block_out, s_.inject, inject_parts_, gamma, s_.xn, n_tokens,
            c.hidden_size, c.hc_count, c.rms_eps, stream_);
}

bool Executor::HcMix(const DeviceMixer& m, const float* res, bool normed,
                     float* mixed, float* inject, std::uint32_t n_tokens,
                     std::string* error_msg) const {
  const Config& c = config();
  // Without `normed` this mixer's grouped norm of res is computed here (F32);
  // otherwise the previous combine produced it, as F16 plus tiled Q8 on the
  // wide route or as F32 in s_.xn.
  if (!normed) {
    xn_half_ = false;
    RmsNormRows(res, m.norm.f32(), s_.xn, n_tokens, c.HcDim(), c.hc_count,
                c.rms_eps, stream_);
  }
  const bool extras = n_tokens <= options_.max_batch &&
                      c.hidden_size <= model_->max_half_cols();
  const bool fused_projection =
      xn_half_ && n_tokens >= 96 && extras && c.hc_count == 4 &&
      c.hidden_size == 2560 && c.hc_low_rank == 320 &&
      m.up.type == GgmlType::kQ8_0 && m.up.rows == c.HcDim() &&
      m.up.cols == c.hc_low_rank;
  if (fused_projection) {
    // The up projection writes x_half; keep its input in the gate buffer.
    if (!HcDownF16Gemm(m.down.data, s_.xn_q8t,
                       reinterpret_cast<__half*>(s_.hc_gate), n_tokens,
                       stream_)) {
      AssignError(error_msg, "fused HC down projection failed");
      return false;
    }
  } else {
    if (xn_half_) {
      if (!W8A8Gemm(m.down.data, s_.xn_q8t, s_.lo, n_tokens, m.down.rows,
                    m.down.cols, stream_)) {
        AssignError(error_msg, "W8A8 mixer down projection failed");
        return false;
      }
    } else if (!Dense(m.down, s_.xn, s_.lo, n_tokens, error_msg)) {
      return false;
    }
    SiluScale(s_.lo, 1.0F / static_cast<float>(c.hc_count),
              static_cast<std::size_t>(n_tokens) * c.hc_low_rank, stream_);
    if (!Dense(m.up, s_.lo, s_.hc_gate, n_tokens, error_msg)) {
      return false;
    }
  }
  const bool fused_inject =
      inject != nullptr && !m.inject.empty() && m.inject.type == GgmlType::kF32;
  const float* xn = s_.xn;
  const bool vectorized =
      xn_half_ ||
      (MatrixRows(n_tokens) && c.hc_count == 4 && c.hidden_size % 4 == 0);
  // `mixed` is being rewritten: whatever the input caches held of it is
  // stale. The wide F16 route also emits the F16 and tiled Q8 copies the
  // projections that follow read.
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  if (xn_half_) {
    if (fused_projection) {
      if (!HcMixF16Gemm(m.up.data, reinterpret_cast<const __half*>(s_.hc_gate),
                        s_.xn_half, fused_inject ? m.inject.f32() : nullptr,
                        mixed, static_cast<__half*>(s_.x_half), s_.x_q8t,
                        inject, n_tokens, c.hidden_size, c.hc_low_rank,
                        stream_)) {
        AssignError(error_msg, "fused HC projection failed");
        return false;
      }
    } else {
      HcMixEpilogueVec4F16(s_.xn_half, s_.hc_gate,
                           fused_inject ? m.inject.f32() : nullptr, mixed,
                           extras ? static_cast<__half*>(s_.x_half) : nullptr,
                           extras ? s_.x_q8t : nullptr, inject, n_tokens,
                           c.hidden_size, stream_);
    }
    if (extras) {
      half_src_ = mixed;
      half_rows_ = n_tokens;
      half_cols_ = c.hidden_size;
      half_bf16_ = false;
      q8t_src_ = mixed;
      q8t_rows_ = n_tokens;
      q8t_cols_ = c.hidden_size;
    }
  } else if (vectorized) {
    HcMixEpilogueVec4(xn, s_.hc_gate, fused_inject ? m.inject.f32() : nullptr,
                      mixed, inject, n_tokens, c.hidden_size, c.hc_count,
                      stream_);
  } else {
    HcMixEpilogue(xn, s_.hc_gate, fused_inject ? m.inject.f32() : nullptr,
                  mixed, inject, n_tokens, c.hidden_size, c.hc_count, stream_);
  }
  inject_parts_ = fused_inject ? (vectorized ? HcInjectPartsVec4(c.hidden_size)
                                             : HcInjectParts(c.hidden_size))
                               : 1;
  if (inject != nullptr && !m.inject.empty() && !fused_inject) {
    // A quantized inject projection (the draft block's) reads an F32 norm;
    // on the wide route the combine only produced F16, so norm again.
    if (xn_half_) {
      RmsNormRows(res, m.norm.f32(), s_.xn, n_tokens, c.HcDim(), c.hc_count,
                  c.rms_eps, stream_);
    }
    if (!Dense(m.inject, xn, inject, n_tokens, error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::PleFetch(Session& session, std::span<const std::int32_t> tokens,
                        bool speculative, std::string* error_msg) const {
  if (ple_pending_ && !WaitPle(error_msg)) {
    return false;
  }
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  // Only proper prefixes need snapshots; the full batch keeps its live state.
  if (speculative) {
    for (std::uint32_t i = 0; i < n; ++i) {
      HashNgramRows(c, session.ngram_, tokens.subspan(i, 1),
                    std::span<std::uint32_t>(
                        host_rows_.data() + i * c.ple_heads, c.ple_heads));
      if (i + 1 < n) {
        session.ngram_snapshots_[i] = session.ngram_;
      }
    }
  } else {
    HashNgramRows(c, session.ngram_, tokens,
                  std::span<std::uint32_t>(host_rows_.data(), n * c.ple_heads));
  }
  ple_pending_ = ngram_->StartRead(
      std::span<const std::uint32_t>(host_rows_.data(), n * c.ple_heads),
      std::span<float>(host_emb_,
                       static_cast<std::size_t>(n) * c.PleEmbeddingDim()));
  if (!ple_pending_) {
    AssignError(error_msg, "n-gram table read could not start");
  }
  return ple_pending_;
}

bool Executor::WaitPle(std::string* error_msg) const {
  const bool ok = ple_pending_ && ngram_->WaitRead();
  ple_pending_ = false;
  if (!ok) {
    AssignError(error_msg, "n-gram table read failed");
  }
  return ok;
}

bool Executor::Ple(const DeviceLayer& l, Session& session, std::uint32_t n,
                   float* res, bool speculative, std::string* error_msg,
                   bool embeddings_ready) const {
  const Config& c = config();
  const std::size_t emb_count =
      static_cast<std::size_t>(n) * c.PleEmbeddingDim();
  if (!embeddings_ready &&
      (!WaitPle(error_msg) ||
       !Check(hipMemcpyAsync(s_.ple_emb, host_emb_, emb_count * sizeof(float),
                             hipMemcpyHostToDevice, stream_),
              "n-gram upload", error_msg))) {
    return false;
  }
  const std::uint32_t hc_dim = c.HcDim();
  Q8Input emb;
  if (!Quantize(s_.ple_emb, n, c.PleEmbeddingDim(), &emb, error_msg) ||
      !Dense(l.ple_key, emb, s_.ple_key, error_msg) ||
      !Dense(l.ple_value, emb, s_.ple_value, error_msg)) {
    return false;
  }
  RmsNormRows(s_.ple_key, l.ple_norm_key.f32(), s_.ple_key, n, hc_dim,
              c.hc_count, c.rms_eps, stream_);
  RmsNormRows(res, l.ple_norm_query.f32(), s_.ple_query, n, hc_dim, c.hc_count,
              c.rms_eps, stream_);
  PleGate(s_.ple_key, s_.ple_query, s_.ple_value, s_.ple_gated, n,
          c.hidden_size, c.hc_count, stream_);
  RmsNormRows(s_.ple_gated, l.ple_norm_conv.f32(), s_.ple_norm, n, hc_dim,
              c.hc_count, c.rms_eps, stream_);
  PleConv(s_.ple_norm, l.ple_conv1d.f32(), session.ple_history_,
          s_.ple_history_scratch, s_.ple_conv,
          speculative ? session.ple_snapshots_ : RollbackRows{}, n, hc_dim,
          c.ple_conv_kernel, c.ple_ngram_size, stream_);
  PleInject(res, s_.ple_gated, s_.ple_conv,
            static_cast<std::size_t>(n) * hc_dim, stream_);
  return true;
}

bool Executor::LinearAttention(const DeviceLayer& l, Session::LinearState& s,
                               const float* x, float* out,
                               std::uint32_t n_tokens, bool speculative,
                               std::string* error_msg, bool projections_ready,
                               bool project_output) const {
  const Config& c = config();
  const std::uint32_t channels = c.SsmConvChannels();
  const float* qkv = s_.qkv;
  const float* z = s_.z;
  std::uint32_t qkv_stride = channels;
  std::uint32_t z_stride = c.SsmValueDim();
  bool convolved = false;
  if (!l.ssm_in.empty()) {
    // Wide prefill convolves QKV in the projection's LDS tile. The raw
    // boundary rows remain available for the rolling history update.
    if (!projections_ready && !speculative && n_tokens >= 1024 &&
        n_tokens <= options_.max_batch && DenseF16Route(l.ssm_in, n_tokens)) {
      PrepareHalfInput(x, n_tokens, l.ssm_in.cols);
      convolved = DenseF16SsmGemm(
          l.ssm_in.data, static_cast<const __half*>(s_.x_half),
          l.ssm_conv1d.f32(), s.conv_state, s_.qkvz, s_.conv_scratch, n_tokens,
          l.ssm_in.rows, l.ssm_in.cols, channels, c.ssm_conv_kernel, stream_);
    }
    if (!projections_ready && !convolved &&
        !Dense(l.ssm_in, x, s_.qkvz, n_tokens, error_msg)) {
      return false;
    }
    qkv = s_.qkvz;
    z = s_.qkvz + channels;
    qkv_stride = z_stride = l.ssm_in.rows;
  } else if (!projections_ready) {
    Q8Input xq;
    if (!Quantize(x, n_tokens, c.hidden_size, &xq, error_msg) ||
        !Dense(l.ssm_qkv, xq, s_.qkv, error_msg) ||
        !Dense(l.ssm_gate, xq, s_.z, error_msg)) {
      return false;
    }
  }
  if (!projections_ready &&
      !Dense(l.ssm_alpha_beta, x, s_.alpha_beta, n_tokens, error_msg)) {
    return false;
  }
  // Keep the same activation precision across prefill chunk boundaries.
  // The F16 epilogue reuses the F32 output allocation. Switching this
  // projection to Q8 for a short tail changes every subsequent layer.
  const bool tiled = MatrixRows(n_tokens) &&
                     l.ssm_out.type == GgmlType::kQ8_0 &&
                     n_tokens <= options_.max_batch;
  const bool half_output =
      tiled && l.ssm_out.rows == 2560 && l.ssm_out.cols == 6144;
  auto* out_half =
      half_output ? reinterpret_cast<__half*>(s_.gdn_out) : nullptr;
  GatedDeltaNet(qkv, qkv_stride, z, z_stride, s_.alpha_beta, l.ssm_conv1d.f32(),
                l.ssm_a.f32(), l.ssm_dt.f32(), l.ssm_norm.f32(), s.conv_state,
                s_.conv_scratch, s_.qn, s_.kn, s_.gdn_raw, s.state, s_.gdn_out,
                tiled && !half_output ? s_.x_q8t : nullptr,
                speculative ? s.state_snapshots : RollbackRows{},
                speculative ? s.conv_snapshots : RollbackRows{}, n_tokens,
                c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_head_dim,
                c.ssm_conv_kernel, MatrixRows(n_tokens) && !speculative,
                convolved, c.rms_eps, stream_, out_half);
  if (!project_output) {
    return true;
  }
  if (half_output) {
    if (!DenseF16Gemm(l.ssm_out.data, out_half, out, n_tokens, l.ssm_out.rows,
                      l.ssm_out.cols, stream_)) {
      AssignError(error_msg, "SSM output F16 GEMM failed");
      return false;
    }
    return true;
  }
  if (tiled) {
    q8t_src_ = nullptr;
    if (!W8A8Gemm(l.ssm_out.data, s_.x_q8t, out, n_tokens, l.ssm_out.rows,
                  l.ssm_out.cols, stream_)) {
      AssignError(error_msg, "W8A8 GEMM failed");
      return false;
    }
    return true;
  }
  return Dense(l.ssm_out, s_.gdn_out, out, n_tokens, error_msg);
}

bool Executor::Attention(const DeviceLayer& l, Session::AttentionState& s,
                         const float* x, float* out, std::uint32_t n_tokens,
                         const std::uint32_t* pos,
                         const std::uint32_t* first_block,
                         std::uint32_t start_pos, std::uint32_t pool_grid,
                         std::uint32_t max_context, bool sparse,
                         std::string* error_msg, bool last_only,
                         bool projections_ready, bool project_output) const {
  const Config& c = config();
  const std::uint32_t kv_row = c.AttentionKvDim();
  const std::uint32_t index_capacity =
      IndexerCapacity(c, options_.max_batch, max_context);
  bool prepared = false;
  if (!l.attn_qkv.empty()) {
    const bool fused_projection =
        !projections_ready && n_tokens >= 1024 &&
        n_tokens <= options_.max_batch && DenseF16Route(l.attn_qkv, n_tokens) &&
        l.attn_qkv.rows == 13312 && l.attn_qkv.cols == 2560 &&
        c.num_heads == 24 && c.num_kv_heads == 2 && c.head_dim == 256 &&
        c.rotary_dim == 64;
    if (fused_projection) {
      PrepareHalfInput(x, n_tokens, l.attn_qkv.cols);
      prepared = AttentionF16Gemm(
          l.attn_qkv.data, static_cast<const __half*>(s_.x_half),
          l.attn_q_norm.f32(), l.attn_k_norm.f32(), s_.q, s_.attn_gate,
          s.k_cache, s.v_cache, n_tokens, pos, c.rope_theta, c.rms_eps, stream_,
          s.rope);
      if (!prepared) {
        AssignError(error_msg, "fused attention projection failed");
        return false;
      }
    } else {
      if (!projections_ready &&
          !Dense(l.attn_qkv, x, s_.qg, n_tokens, error_msg)) {
        return false;
      }
      prepared = PrepareAttention(
          s_.qg, l.attn_qkv.rows, l.attn_q_norm.f32(), l.attn_k_norm.f32(),
          s_.q, s_.attn_gate, s.k_cache, s.v_cache, n_tokens, c.num_heads,
          c.num_kv_heads, c.head_dim, c.rotary_dim, pos, c.rope_theta,
          c.rms_eps, stream_, s.rope, prefill_phase);
      if (!prepared) {
        UnpackQGate(s_.qg, l.attn_qkv.rows, s_.q, s_.attn_gate, s_.k, s_.v,
                    n_tokens, c.num_heads, c.head_dim, kv_row, stream_);
      }
    }
  } else {
    Q8Input xq;
    if (!projections_ready &&
        (!Quantize(x, n_tokens, c.hidden_size, &xq, error_msg) ||
         !Dense(l.attn_q, xq, s_.qg, error_msg) ||
         !Dense(l.attn_k, xq, s_.k, error_msg) ||
         !Dense(l.attn_v, xq, s_.v, error_msg))) {
      return false;
    }
    UnpackQGate(s_.qg, 2 * c.AttentionQDim(), s_.q, s_.attn_gate, nullptr,
                nullptr, n_tokens, c.num_heads, c.head_dim, 0, stream_);
  }
  if (!prepared) {
    RmsNormRows(s_.q, l.attn_q_norm.f32(), s_.q, n_tokens * c.num_heads,
                c.head_dim, 1, c.rms_eps, stream_);
    RmsNormRows(s_.k, l.attn_k_norm.f32(), s_.k, n_tokens * c.num_kv_heads,
                c.head_dim, 1, c.rms_eps, stream_);
    Rope(s_.q, n_tokens, c.num_heads, c.head_dim, c.rotary_dim, pos,
         c.rope_theta, stream_, s.rope);
    Rope(s_.k, n_tokens, c.num_kv_heads, c.head_dim, c.rotary_dim, pos,
         c.rope_theta, stream_, s.rope);
    StoreKv(s_.k, s.k_cache, n_tokens, kv_row, pos, stream_);
    StoreKv(s_.v, s.v_cache, n_tokens, kv_row, pos, stream_);
  }

  // Raw keys survive only until pooling. The ring holds the initial
  // pre-budget backlog and a batch, independently for trunk and predictor.
  if (s.index_k != nullptr) {
    if (!Dense(l.indexer_k, x, s_.ik, n_tokens, error_msg)) {
      return false;
    }
    StoreRows(s_.ik, s.index_k, n_tokens, c.indexer_head_dim, pos,
              index_capacity, stream_);
  }
  const std::uint32_t* mask = nullptr;
  if (sparse) {
    if (!Dense(l.indexer_q, x, s_.iq, n_tokens, error_msg)) {
      return false;
    }
    RmsNormRows(s_.iq, l.indexer_q_norm.f32(), s_.iq,
                n_tokens * c.indexer_heads, c.indexer_head_dim, 1, c.rms_eps,
                stream_);
    Rope(s_.iq, n_tokens, c.indexer_heads, c.indexer_head_dim, c.rotary_dim,
         pos, c.rope_theta, stream_, s.rope);
    PoolIndexerBlocks(s.index_k, l.indexer_k_norm.f32(), s.block_k, first_block,
                      pos, n_tokens, pool_grid, c.compress_ratio,
                      c.indexer_head_dim, c.rotary_dim, c.rope_theta, c.rms_eps,
                      index_capacity, stream_, s.rope);
    // Align score rows to full cache lines. The selector still considers
    // only complete causal blocks, so padding cannot change the ranking.
    const std::uint32_t blocks =
        (max_context + c.compress_ratio - 1) / c.compress_ratio;
    const std::uint32_t max_blocks = (blocks + 31) / 32 * 32;
    // Catch-up consumes only the final attention tile. Keep all its query
    // masks (sparse attention packs four queries; dense tiles hold sixteen)
    // but avoid scoring the unused prefix against the complete context.
    const auto first_query = last_only ? (n_tokens - 1) / 16 * 16 : 0U;
    for (std::uint32_t t0 = first_query; t0 < n_tokens; t0 += select_chunk_) {
      const std::uint32_t n = std::min(select_chunk_, n_tokens - t0);
      SelectBlocks(s_.iq + static_cast<std::size_t>(t0) * c.indexer_heads *
                               c.indexer_head_dim,
                   s.block_k,
                   s_.mask + static_cast<std::size_t>(t0) * mask_words_,
                   s_.scores, n, pos, t0, c.indexer_heads, c.indexer_head_dim,
                   c.compress_ratio, c.indexer_top_k / c.compress_ratio,
                   mask_words_, max_blocks, stream_);
    }
    mask = s_.mask;
  }
  // Wide batches run the fused WMMA kernel (never inside a graph: the kv
  // extent is a host value), output gate included, skipping the key tiles
  // no query of a block selected; the per-token kernel covers the rest.
  if (last_only && !Check(hipMemsetAsync(s_.ctx, 0,
                                         static_cast<std::size_t>(n_tokens) *
                                             c.AttentionQDim() * sizeof(float),
                                         stream_),
                          "draft attention output initialization", error_msg)) {
    return false;
  }
  if (MatrixRows(n_tokens) &&
      WmmaCausalAttention(s_.q, s_.attn_gate, s.k_cache, s.v_cache, mask,
                          mask_words_, s_.ctx, n_tokens, start_pos, c.num_heads,
                          c.num_kv_heads, c.head_dim, c.compress_ratio, stream_,
                          last_only)) {
    return !project_output ||
           Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
  }
  // Narrow batches split each row's key tiles over kAttnSplits blocks so a
  // decode step at depth fills the device.
  const bool split = !MatrixRows(n_tokens);
  rocm::Attention(s_.q, s.k_cache, s.v_cache, mask, mask_words_, s_.ctx,
                  split ? s_.attn_partials : nullptr, kAttnSplits, n_tokens,
                  pos, c.num_heads, c.num_kv_heads, c.head_dim,
                  c.compress_ratio, stream_);
  SigmoidMul(s_.ctx, s_.attn_gate,
             static_cast<std::size_t>(n_tokens) * c.AttentionQDim(), stream_);
  return !project_output || Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg);
}

bool Executor::Moe(const DeviceLayer& l, const float* x, float* out,
                   std::uint32_t n_tokens, std::string* error_msg,
                   bool last_only) const {
  const Config& c = config();
  const std::uint32_t used = c.num_experts_used;
  // Router logits and the shared-expert gate come out of one GEMM.
  if (!Dense(l.router, x, s_.router, n_tokens, error_msg)) {
    return false;
  }
  RouterTopK(s_.router, c.num_experts + 1, s_.ids, s_.weights, n_tokens,
             c.num_experts, used, stream_);
  if (ExpertMatrixRows(n_tokens)) {
    ExpertCounts(s_.ids, s_.expert_counts, n_tokens, c.num_experts, used,
                 stream_);
    if (!Check(hipMemcpyAsync(counts_host_, s_.expert_counts,
                              c.num_experts * sizeof(std::uint32_t),
                              hipMemcpyDeviceToHost, stream_),
               "expert counts download", error_msg) ||
        !Check(hipEventRecord(counts_ready_, stream_), "expert counts event",
               error_msg)) {
      return false;
    }
  }
  // The shared expert (gated by the last router row) does not depend on
  // routing. Queue its GEMMs after the count download, then prepare the
  // routed dispatch on the CPU without waiting for these GEMMs to finish.
  shexp_half_ready_ = false;
  if (!GatedDense(l.shexp_up, l.shexp_gate, x, s_.shexp_up, n_tokens,
                  &l.shexp_down, error_msg)) {
    return false;
  }
  if (shexp_half_ready_) {
    shexp_half_ready_ = false;
    if (!DenseF16Gemm(l.shexp_down.data, s_.shexp_half, s_.shexp_out, n_tokens,
                      l.shexp_down.rows, l.shexp_down.cols, stream_)) {
      AssignError(error_msg, "shared expert F16 GEMM failed");
      return false;
    }
  } else if (!Dense(l.shexp_down, s_.shexp_up, s_.shexp_out, n_tokens,
                    error_msg)) {
    return false;
  }
  if (last_only && l.ffn_gate_exps.type == GgmlType::kQ8_0 &&
      l.ffn_up_exps.type == GgmlType::kQ8_0 &&
      l.ffn_down_exps.type == GgmlType::kQ8_0) {
    // Catch-up retains only the final predictor residual. Keep the router
    // and shared-expert shapes, then run its routed experts with the same
    // tiled quantization and accumulation as the complete prompt batch.
    const auto base = s_;
    UseScratch(RowScratch(base, n_tokens - 1));
    PrefillPhase phase(true);
    try {
      const std::size_t offset =
          static_cast<std::size_t>(n_tokens - 1) * c.hidden_size;
      const bool ok = MoeExperts(l, x + offset, out + offset, 1, error_msg);
      UseScratch(base);
      return ok;
    } catch (...) {
      UseScratch(base);
      throw;
    }
  }
  return MoeExperts(l, x, out, n_tokens, error_msg);
}

bool Executor::MoeExperts(const DeviceLayer& l, const float* x, float* out,
                          std::uint32_t n_tokens,
                          std::string* error_msg) const {
  const Config& c = config();
  const std::uint32_t used = c.num_experts_used;
  const std::uint32_t slots = n_tokens * used;
  if (!RouteHints(n_tokens, error_msg)) {
    return false;
  }
  // Tiled batches take the WMMA route: assignments compacted by expert
  // into 16-row padded buckets, token rows narrowed to F16 once, then the
  // F16 matrix-core GEMM per (expert, row tile).
  const bool wmma_experts = ExpertMatrixRows(n_tokens) &&
                            (l.ffn_gate_exps.type == GgmlType::kQ4_K ||
                             l.ffn_gate_exps.type == GgmlType::kQ5_K) &&
                            l.ffn_up_exps.type == l.ffn_gate_exps.type &&
                            (l.ffn_down_exps.type == GgmlType::kQ5_1 ||
                             l.ffn_down_exps.type == GgmlType::kQ8_0) &&
                            c.hidden_size % 256 == 0 && c.expert_ff % 64 == 0;
  if (wmma_experts) {
    RoutedCompact(s_.ids, s_.expert_counts, s_.routed_bounds, s_.routed_cursors,
                  s_.rows_token, s_.rows_slot, n_tokens, used, c.num_experts,
                  stream_);
    // The GEMMs read F16 token rows: the router's F16 GEMM (or the mix)
    // usually left them in s_.x_half already.
    if (!(half_src_ == x && half_rows_ == n_tokens &&
          half_cols_ == c.hidden_size && !half_bf16_)) {
      NarrowActivations(x, s_.x_half, false,
                        static_cast<std::size_t>(n_tokens) * c.hidden_size,
                        stream_);
      half_src_ = x;
      half_rows_ = n_tokens;
      half_cols_ = c.hidden_size;
      half_bf16_ = false;
    }
    const auto* x_half = static_cast<const __half*>(s_.x_half);
    // Large batches pair the gate/up projections and apply SwiGLU without
    // materializing the gate. Smaller buckets favor separate projections.
    auto* up_half = reinterpret_cast<__half*>(s_.up_e);
    const WeightType gate_type = l.ffn_gate_exps.type == GgmlType::kQ5_K
                                     ? WeightType::kQ5_K
                                     : WeightType::kQ4_K;
    const bool gated_ok =
        n_tokens >= 1024 && routed_tile_rows_ == 48
            ? RoutedGatedF16Gemm(
                  l.ffn_gate_exps.data, l.ffn_up_exps.data, gate_type, x_half,
                  s_.routed_tiles + routed_pair_offset_, routed_pair_tiles_,
                  routed_pair_rows_, s_.routed_bounds, s_.rows_token,
                  s_.rows_slot, up_half, c.expert_ff, c.hidden_size, stream_)
            : (RoutedF16Gemm(l.ffn_gate_exps.data, gate_type, x_half,
                             s_.routed_tiles, routed_n_tiles_,
                             routed_tile_rows_, s_.routed_bounds, s_.rows_token,
                             s_.rows_slot, nullptr, s_.gate_e, nullptr,
                             c.expert_ff, c.hidden_size, stream_) &&
               RoutedF16Gemm(l.ffn_up_exps.data, gate_type, x_half,
                             s_.routed_tiles, routed_n_tiles_,
                             routed_tile_rows_, s_.routed_bounds, s_.rows_token,
                             s_.rows_slot, s_.gate_e, nullptr, up_half,
                             c.expert_ff, c.hidden_size, stream_));
    if (!gated_ok) {
      AssignError(error_msg, "routed F16 gate/up GEMM failed");
      return false;
    }
    const WeightType down_type = l.ffn_down_exps.type == GgmlType::kQ8_0
                                     ? WeightType::kQ8_0
                                     : WeightType::kQ5_1;
    // Larger down tiles amortize weight decoding. Reuse the 64-token map
    // when it has no more padded rows than the 48-token map.
    const bool wide_down =
        (down_type == WeightType::kQ5_1 || down_type == WeightType::kQ8_0) &&
        routed_tile_rows_ == 48 && routed_64_tiles_ != 0 &&
        routed_64_tiles_ * 4 <= routed_n_tiles_ * 3;
    // The down projection's rows are F16 too: the epilogue reads half the
    // bytes of the largest routed intermediate.
    if (!RoutedF16Gemm(l.ffn_down_exps.data, down_type, up_half,
                       s_.routed_tiles + (wide_down ? routed_n_tiles_ : 0),
                       wide_down ? routed_64_tiles_ : routed_n_tiles_,
                       wide_down ? 64 : routed_tile_rows_, s_.routed_bounds,
                       s_.rows_slot, s_.rows_slot, nullptr, nullptr,
                       reinterpret_cast<__half*>(s_.down_e), c.hidden_size,
                       c.expert_ff, stream_)) {
      AssignError(error_msg, "routed F16 down GEMM failed");
      return false;
    }
  } else {
    if (!GatedExperts(l.ffn_gate_exps, l.ffn_up_exps, x, s_.ids, s_.gate_e,
                      n_tokens, used, error_msg)) {
      return false;
    }
    // The down projection sees one (token, slot) row per expert id.
    if (!Experts(l.ffn_down_exps, s_.gate_e, s_.ids, s_.down_e, slots, 1,
                 n_tokens, error_msg)) {
      return false;
    }
  }
  if (wmma_experts) {
    // The combine that follows folds this epilogue into its own pass when
    // it takes the F16 route (Combine); otherwise it runs here.
    moe_pending_ = out == s_.block_out;
    if (!moe_pending_) {
      MoeEpilogueVec4F16(reinterpret_cast<const __half*>(s_.down_e), s_.weights,
                         s_.shexp_out, s_.router + c.num_experts,
                         c.num_experts + 1, out, n_tokens, used, c.hidden_size,
                         stream_);
    }
  } else if (MatrixRows(n_tokens)) {
    MoeEpilogueVec4(s_.down_e, s_.weights, s_.shexp_out,
                    s_.router + c.num_experts, c.num_experts + 1, out, n_tokens,
                    used, c.hidden_size, stream_);
  } else {
    MoeEpilogue(s_.down_e, s_.weights, s_.shexp_out, s_.router + c.num_experts,
                c.num_experts + 1, out, n_tokens, used, c.hidden_size, stream_);
  }
  return true;
}

bool Executor::CopyTrunkHidden(const Session& session, std::span<float> hidden,
                               std::string* error_msg) const {
  if (!session.mtp_enabled_ || session.owner_ != this ||
      session.position_ == 0 || hidden.empty() ||
      hidden.size() % config().HcDim() != 0 ||
      hidden.size() / config().HcDim() >
          std::min(session.position_, options_.max_speculative)) {
    AssignError(error_msg, "invalid trunk hidden diagnostic input");
    return false;
  }
  return Check(hipMemcpyAsync(hidden.data(), session.mtp_.target_hidden,
                              hidden.size_bytes(), hipMemcpyDeviceToHost,
                              stream_),
               "trunk hidden download", error_msg) &&
         Check(hipStreamSynchronize(stream_), "trunk hidden", error_msg);
}

bool Executor::MtpHead(const DeviceMixer& head, const float* res, bool token,
                       bool candidates, std::string* error_msg) const {
  const DeviceTensor& output = model_->output();
  if (!HcMix(head, res, false, s_.mixed, nullptr, 1, error_msg) ||
      !Dense(output, s_.mixed, s_.logits, 1, error_msg)) {
    return false;
  }
  if (candidates)
    MtpTopCandidates(s_.logits, s_.mtp_ids, s_.mtp_scratch_ids, s_.mtp_scores,
                     output.rows, stream_);
  if (token) {
    Argmax(s_.logits, s_.mtp_argmax, s_.mtp_token, 1, output.rows, stream_);
    if (!Check(
            hipMemcpyAsync(mtp_token_host_, s_.mtp_token, sizeof(std::int32_t),
                           hipMemcpyDeviceToHost, stream_),
            "draft token download", error_msg)) {
      return false;
    }
  }
  if (candidates) {
    const auto count = std::min<std::size_t>(output.rows, kMtpCandidates);
    static_assert(offsetof(MtpCandidateLogits, logits) ==
                  kMtpCandidates * sizeof(std::uint32_t));
    const auto bytes =
        offsetof(MtpCandidateLogits, logits) + count * sizeof(float);
    if (!Check(hipMemcpyAsync(mtp_candidates_host_, s_.mtp_ids, bytes,
                              hipMemcpyDeviceToHost, stream_),
               "draft candidates download", error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::Run(Session& session, std::uint64_t key, bool graph,
                   const std::function<bool()>& body, std::string* error_msg,
                   bool synchronize) const {
  // A batch shape runs eagerly once before it is captured: the first pass
  // grows the GEMM tier's arena, which capture forbids.
  if (graph && session.warmed_.contains(key)) {
    hipGraphExec_t exec = nullptr;
    if (const auto it = session.graphs_.find(key);
        it != session.graphs_.end()) {
      exec = it->second;
    } else {
      hipGraph_t captured = nullptr;
      // Frozen peer sessions can copy snapshots on independent nonblocking
      // streams while the scheduler records this session's decode graph.
      if (!Check(
              hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal),
              "graph capture", error_msg)) {
        return false;
      }
      const bool ok = body();
      if (!Check(hipStreamEndCapture(stream_, &captured), "graph capture end",
                 error_msg) ||
          !ok) {
        if (captured != nullptr) {
          (void)hipGraphDestroy(captured);
        }
        return false;
      }
      const bool instantiated =
          Check(hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0),
                "graph instantiate", error_msg);
      (void)hipGraphDestroy(captured);
      if (!instantiated) {
        return false;
      }
      session.graphs_.emplace(key, exec);
    }
    if (!Check(hipGraphLaunch(exec, stream_), "graph launch", error_msg)) {
      return false;
    }
  } else {
    if (!body()) {
      return false;
    }
    session.warmed_.insert(key);
  }
  return !synchronize ||
         Check(hipStreamSynchronize(stream_), "forward", error_msg);
}

bool Executor::Forward(Session& session, std::span<const std::int32_t> tokens,
                       std::uint32_t n_logits, float* logits, ForwardMode mode,
                       std::string* error_msg) const {
  const bool speculative = mode == ForwardMode::kVerify;
  PrefillPhase phase(mode == ForwardMode::kPrefill);
  selected_logits_ = nullptr;
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > options_.max_batch ||
      (speculative && n > options_.max_speculative)) {
    AssignError(error_msg, "token batch is empty or exceeds the batch limit");
    return false;
  }
  if (n_logits > n || n_logits > options_.max_logit_rows) {
    AssignError(error_msg, "requested logit rows exceed the batch or limit");
    return false;
  }
  if (session.owner_ != this || session.position_ + n > session.max_context_) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  for (auto t : tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= c.vocab_size) {
      AssignError(error_msg, "token out of range");
      return false;
    }
  }
  const std::uint32_t start_pos = session.position_;
  if (!session.CheckCancellation(error_msg))
    return false;
  if (session.mtp_enabled_ && prefill_phase &&
      session.mtp_.position != start_pos) {
    AssignError(error_msg,
                "MTP must consume the preceding frontier before prefill");
    return false;
  }
  if (speculative && !EnsureRollback(session, n - 1, error_msg))
    return false;
  ++session.mutation_epoch_;
  session.spec_base_ = start_pos;
  session.spec_tokens_ = speculative ? n : 0;
  // Everything the launched work reads from the host sits in pinned
  // buffers the graph nodes point at: tokens, the control block (positions
  // the kernels read on the device) and the n-gram rows.
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = start_pos;
  control_host_->blocks = session.blocks_;
  control_host_->mtp_position = session.mtp_.position;
  control_host_->mtp_blocks = session.mtp_.blocks;
  control_host_->hidden_row = -1;
  if (c.ple_layer >= 0) {
    if (ngram_ == nullptr) {
      AssignError(error_msg, "n-gram table is not open");
      return false;
    }
    if (!PleFetch(session, tokens, speculative, error_msg)) {
      return false;
    }
  }
  // Sparse selection only changes the result once a query can see more
  // than the token budget; every layer of this model shares one ratio.
  const bool sparse = c.compress_ratio > 0 && start_pos + n > c.indexer_top_k;
  const std::uint32_t complete =
      c.compress_ratio > 0 ? (start_pos + n) / c.compress_ratio : 0;
  const std::uint32_t pool_grid =
      sparse && complete > session.blocks_ ? complete - session.blocks_ : 0;
  // Decode-sized batches replay as graphs; a pooling backlog (the first
  // batch past the budget) needs the wider eager grid.
  const std::uint32_t graph_pool_grid = n / std::max(c.compress_ratio, 1u) + 1;
  const bool graph = !prefill_phase && n <= kVecBatch &&
                     pool_grid <= graph_pool_grid &&
                     session.position_ >= session.VisionLayout().PrefixLength();
  const std::uint64_t key = static_cast<std::uint64_t>(n) |
                            (static_cast<std::uint64_t>(n_logits) << 16) |
                            (static_cast<std::uint64_t>(speculative) << 32) |
                            (static_cast<std::uint64_t>(sparse) << 33) |
                            (std::uint64_t{logits != nullptr} << 34);
  // PLE first consumes the disk rows at its injection layer. Queue the
  // preceding layers before waiting, including on captured graph replay.
  // Both pieces use the same stream and arithmetic as the unsplit graph.
  const std::uint32_t first_layer =
      graph && c.ple_layer > 0 ? static_cast<std::uint32_t>(c.ple_layer) : 0;
  if (first_layer > 0) {
    const auto prefix = [&] {
      return ForwardBody(session, n, 0, false, speculative, sparse, start_pos,
                         graph_pool_grid, 0, first_layer, error_msg);
    };
    constexpr std::uint64_t kPrefixKey = std::uint64_t{1} << 35;
    if (!Run(session, key | kPrefixKey, graph, prefix, error_msg, false)) {
      return false;
    }
  }
  const auto body = [&] {
    return ForwardBody(session, n, n_logits, logits != nullptr, speculative,
                       sparse, start_pos, graph ? graph_pool_grid : pool_grid,
                       first_layer, c.num_layers, error_msg);
  };
  // Only the suffix waits for its pinned n-gram rows. During eager
  // execution and capture, Ple performs this wait at the same boundary.
  if (graph && session.graphs_.contains(key) && ple_pending_ &&
      !WaitPle(error_msg)) {
    return false;
  }
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  if (n_logits > 0 && logits != nullptr) {
    std::copy_n(logits_host_, static_cast<std::size_t>(n_logits) * c.vocab_size,
                logits);
  }

  session.position_ += n;
  if (sparse) {
    session.blocks_ = complete;
  }
  if (session.mtp_enabled_ && prefill_phase && n > 1) {
    // Every successor except the last one is already known. Consume it while
    // the trunk residual is still in shared scratch. MtpBody reads that
    // residual into xn before reusing res for the predictor output.
    PrefillPhase draft_phase(false);
    if (!MtpForward(session, tokens.subspan(1), 0, {}, error_msg, s_.res))
      return false;
  }
  return true;
}

bool Executor::ForwardBody(Session& session, std::uint32_t n,
                           std::uint32_t n_logits, bool download_logits,
                           bool speculative, bool sparse,
                           std::uint32_t start_pos, std::uint32_t pool_grid,
                           std::uint32_t first_layer, std::uint32_t end_layer,
                           std::string* error_msg) const {
  const Config& c = config();
  if (first_layer == 0) {
    if (!Check(hipMemcpyAsync(session.control_, control_host_,
                              sizeof(Session::Control), hipMemcpyHostToDevice,
                              stream_),
               "control upload", error_msg) ||
        !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               "token upload", error_msg)) {
      return false;
    }
    EmbedTokens(model_->token_embd().data, SmallType(model_->token_embd().type),
                s_.tokens, s_.res, n, c.hidden_size, c.hc_count, stream_);
    session.vision_input_.Inject(s_.res, start_pos, n, c.hidden_size,
                                 c.hc_count, stream_);
  }
  const auto& layers = model_->layers();
  bool normed = false;  ///< xn holds the next mixer's grouped norm of res
  for (std::uint32_t il = first_layer; il < end_layer; ++il) {
    if (!session.CheckCancellation(error_msg))
      return false;
    const DeviceLayer& l = layers[il];
    if (c.IsPleLayer(il) &&
        !Ple(l, session, n, s_.res, speculative, error_msg)) {
      return false;
    }
    if (!HcMix(l.hc_attn, s_.res, normed, s_.mixed, s_.inject, n, error_msg)) {
      return false;
    }
    if (l.linear) {
      if (!LinearAttention(l, session.linear_[il], s_.mixed, s_.block_out, n,
                           speculative, error_msg)) {
        return false;
      }
    } else if (!Attention(l, session.attention_[il], s_.mixed, s_.block_out, n,
                          &session.control_->position,
                          &session.control_->blocks, start_pos, pool_grid,
                          session.max_context_, sparse, error_msg)) {
      return false;
    }

    // Each combine also norms the residual for the mixer that follows it,
    // unless PLE rewrites the residual first.
    Combine(s_.res, l.hc_ffn.norm.f32(), n);
    if (!HcMix(l.hc_ffn, s_.res, true, s_.mixed, s_.inject, n, error_msg) ||
        !Moe(l, s_.mixed, s_.block_out, n, error_msg)) {
      return false;
    }
    const float* next_norm =
        il + 1 < c.num_layers
            ? (c.IsPleLayer(il + 1) ? nullptr
                                    : layers[il + 1].hc_attn.norm.f32())
            : model_->hc_head().norm.f32();
    Combine(s_.res, next_norm, n);
    normed = next_norm != nullptr;
  }
  if (end_layer < c.num_layers) {
    return true;
  }
  // Prefill catches up immediately from shared scratch. Only the short tail
  // needed across calls belongs to this session.
  const auto kept = std::min(n, options_.max_speculative);
  if (session.mtp_enabled_ &&
      !Check(hipMemcpyAsync(
                 session.mtp_.target_hidden,
                 s_.res + static_cast<std::size_t>(n - kept) * c.HcDim(),
                 static_cast<std::size_t>(kept) * c.HcDim() * sizeof(float),
                 hipMemcpyDeviceToDevice, stream_),
             "hidden keep", error_msg)) {
    return false;
  }
  if (n_logits > 0) {
    PrefillPhase head_phase(false);
    // The head mixer norms its tail rows itself: the last combine's norm is
    // laid out for the whole batch (and tiled on the wide route), so a row
    // offset into it is not addressable.
    const std::size_t skip = static_cast<std::size_t>(n - n_logits);
    const DeviceMixer& head = model_->hc_head();
    if (!HcMix(head, s_.res + skip * c.HcDim(), false, s_.mixed, nullptr,
               n_logits, error_msg) ||
        !Dense(model_->output(), s_.mixed, s_.logits, n_logits, error_msg)) {
      return false;
    }
    if (download_logits &&
        !Check(hipMemcpyAsync(logits_host_, s_.logits,
                              static_cast<std::size_t>(n_logits) *
                                  c.vocab_size * sizeof(float),
                              hipMemcpyDeviceToHost, stream_),
               "logits download", error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::Rollback(Session& session, std::uint32_t keep,
                        std::string* error_msg, float* logits) const {
  const Config& c = config();
  const std::uint32_t n = session.spec_tokens_;
  if (n == 0 || keep == 0 || keep > n) {
    AssignError(error_msg, "rollback outside the pending speculative batch");
    return false;
  }
  session.spec_tokens_ = 0;
  if (logits != nullptr &&
      !Check(hipMemcpyAsync(
                 logits_host_,
                 VerificationLogits() +
                     static_cast<std::size_t>(keep - 1) * c.vocab_size,
                 c.vocab_size * sizeof(float), hipMemcpyDeviceToHost, stream_),
             "frontier download", error_msg)) {
    return false;
  }
  if (keep == n && logits == nullptr) {
    return true;
  }
  if (keep < n) {
    const std::size_t conv_elems =
        static_cast<std::size_t>(c.ssm_conv_kernel - 1) * c.SsmConvChannels();
    const std::size_t slot = keep - 1;
    for (auto& l : session.linear_) {
      if (l.state == nullptr) {
        continue;
      }
      RestoreGdnState(l.state, l.state_snapshots, keep, c.ssm_num_k_heads,
                      c.ssm_num_v_heads, stream_);
      if (!Check(hipGetLastError(), "state rollback", error_msg) ||
          !Check(hipMemcpyAsync(l.conv_state, l.conv_snapshots.rows[slot],
                                conv_elems * sizeof(float),
                                hipMemcpyDeviceToDevice, stream_),
                 "conv rollback", error_msg)) {
        return false;
      }
    }
    if (session.ple_history_ != nullptr) {
      const std::size_t hist =
          static_cast<std::size_t>(c.PleConvHistory()) * c.HcDim();
      if (!Check(hipMemcpyAsync(
                     session.ple_history_, session.ple_snapshots_.rows[slot],
                     hist * sizeof(float), hipMemcpyDeviceToDevice, stream_),
                 "PLE rollback", error_msg)) {
        return false;
      }
      session.ngram_ = session.ngram_snapshots_[slot];
    }
    session.position_ = session.spec_base_ + keep;
    // Pooled block keys past the kept prefix are stale; they are rebuilt
    // from the raw keys when needed.
    session.blocks_ =
        c.compress_ratio == 0
            ? 0
            : std::min(session.blocks_, session.position_ / c.compress_ratio);
  }
  if (!Check(hipStreamSynchronize(stream_), "rollback", error_msg)) {
    return false;
  }
  if (logits != nullptr) {
    std::copy_n(logits_host_, c.vocab_size, logits);
  }
  return true;
}

constexpr std::array<char, 8> kSnapshotMagic{'Q', 'F', 'N', 'S',
                                             'N', 'A', 'P', '4'};

/// Fixed header ahead of the section bytes. It carries every geometry
/// value the section sizes derive from, so a payload of another artifact
/// or executor configuration is rejected before anything is copied.
struct SnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t num_layers;
  std::uint32_t full_attention_interval;
  std::uint32_t conv_elems;
  std::uint32_t state_elems;
  std::uint32_t kv_row;
  std::uint32_t indexer_head_dim;
  std::uint32_t compress_ratio;
  std::uint32_t hc_dim;
  std::uint32_t ple_elems;
  std::uint32_t has_mtp;
  std::uint32_t position;
  std::uint32_t blocks;
  std::uint32_t mtp_position;
  std::uint32_t mtp_blocks;
  std::uint32_t hidden_rows;
  std::uint32_t image_count;
  std::array<std::int32_t, Config::kMaxPleNgram - 1> ngram_prev;
  std::uint64_t payload_bytes;
};
static_assert(std::is_trivially_copyable_v<SnapshotHeader>);

namespace {

SnapshotHeader MakeSnapshotHeader(const Config& c, bool has_mtp,
                                  const Session& session) {
  SnapshotHeader h{};
  h.magic = kSnapshotMagic;
  h.num_layers = c.num_layers;
  h.full_attention_interval = c.full_attention_interval;
  h.conv_elems = (c.ssm_conv_kernel - 1) * c.SsmConvChannels();
  h.state_elems = c.ssm_num_v_heads * c.ssm_head_dim * c.ssm_head_dim;
  h.kv_row = c.AttentionKvDim();
  h.indexer_head_dim = c.indexer_head_dim;
  h.compress_ratio = c.compress_ratio;
  h.hc_dim = c.HcDim();
  h.ple_elems = c.ple_layer >= 0 ? c.PleConvHistory() * c.HcDim() : 0;
  h.has_mtp = has_mtp ? 1 : 0;
  h.position = session.position();
  h.image_count = session.VisionLayout().images.size();
  return h;
}

/// Whether `h` describes this executor's geometry (positions aside).
bool SameGeometry(const SnapshotHeader& h, const SnapshotHeader& mine) {
  return h.magic == mine.magic && h.num_layers == mine.num_layers &&
         h.full_attention_interval == mine.full_attention_interval &&
         h.conv_elems == mine.conv_elems && h.state_elems == mine.state_elems &&
         h.kv_row == mine.kv_row &&
         h.indexer_head_dim == mine.indexer_head_dim &&
         h.compress_ratio == mine.compress_ratio && h.hc_dim == mine.hc_dim &&
         h.ple_elems == mine.ple_elems && h.has_mtp == mine.has_mtp;
}

}  // namespace

template<typename Visit>
std::uint64_t Executor::WalkSnapshot(const SnapshotHeader& h,
                                     const Session* session, Visit&& visit) {
  std::uint64_t offset = sizeof(SnapshotHeader);
  const auto region = [&](void* device, std::uint64_t bytes, const char* what) {
    if (bytes != 0 && !visit(device, offset, bytes, what)) {
      return false;
    }
    offset += bytes;
    return true;
  };
  const auto linear = [&](std::uint32_t il) -> const Session::LinearState* {
    return session != nullptr ? &session->linear_[il] : nullptr;
  };
  const auto attention =
      [&](std::uint32_t il) -> const Session::AttentionState* {
    return session != nullptr ? &session->attention_[il] : nullptr;
  };
  for (std::uint32_t il = 0; il < h.num_layers; ++il) {
    if (((il + 1) % h.full_attention_interval) != 0) {
      const auto* l = linear(il);
      if (!region(l != nullptr ? l->conv_state : nullptr,
                  std::uint64_t{h.conv_elems} * sizeof(float), "conv state") ||
          !region(l != nullptr ? l->state : nullptr,
                  std::uint64_t{h.state_elems} * sizeof(float),
                  "recurrent state")) {
        return 0;
      }
    }
  }
  if (!region(session != nullptr ? session->ple_history_ : nullptr,
              std::uint64_t{h.ple_elems} * sizeof(float), "PLE history")) {
    return 0;
  }
  const std::uint64_t kv_bytes =
      std::uint64_t{h.position} * h.kv_row * sizeof(__half);
  const std::uint64_t index_row_bytes =
      std::uint64_t{h.indexer_head_dim} * sizeof(float);
  const std::uint32_t index_begin = h.blocks * h.compress_ratio;
  const std::uint32_t index_rows = h.position - index_begin;
  const std::uint64_t block_bytes =
      std::uint64_t{h.blocks} * h.indexer_head_dim * sizeof(__half);
  for (std::uint32_t il = 0; il < h.num_layers; ++il) {
    if (((il + 1) % h.full_attention_interval) == 0) {
      const auto* at = attention(il);
      if (!region(at != nullptr ? at->k_cache : nullptr, kv_bytes, "K cache") ||
          !region(at != nullptr ? at->v_cache : nullptr, kv_bytes, "V cache")) {
        return 0;
      }
      // Serialize only unpooled rows, in chronological order. The physical
      // ring may wrap, and a destination session may have another capacity.
      if (session == nullptr) {
        if (!region(nullptr, index_rows * index_row_bytes, "indexer keys"))
          return 0;
      } else {
        const auto first = index_begin & (session->index_capacity_ - 1);
        const auto tail =
            std::min(index_rows, session->index_capacity_ - first);
        if (!region(at->index_k + std::size_t{first} * h.indexer_head_dim,
                    tail * index_row_bytes, "indexer tail") ||
            !region(at->index_k, (index_rows - tail) * index_row_bytes,
                    "indexer wrap"))
          return 0;
      }
      if (!region(at != nullptr ? at->block_k : nullptr, block_bytes,
                  "pooled block keys"))
        return 0;
    }
  }
  if (h.has_mtp != 0) {
    const auto* mtp = session != nullptr ? &session->mtp_ : nullptr;
    const std::uint32_t begin = h.mtp_blocks * h.compress_ratio;
    const std::uint32_t rows = h.mtp_position - begin;
    if (session == nullptr) {
      if (!region(nullptr, rows * index_row_bytes, "draft indexer keys"))
        return 0;
    } else {
      const auto first = begin & (session->index_capacity_ - 1);
      const auto tail = std::min(rows, session->index_capacity_ - first);
      if (!region(mtp->index_k + std::size_t{first} * h.indexer_head_dim,
                  tail * index_row_bytes, "draft indexer tail") ||
          !region(mtp->index_k, (rows - tail) * index_row_bytes,
                  "draft indexer wrap"))
        return 0;
    }
    if (!region(
            mtp ? mtp->block_k : nullptr,
            std::uint64_t{h.mtp_blocks} * h.indexer_head_dim * sizeof(__half),
            "draft pooled keys"))
      return 0;
    const std::uint64_t mtp_kv_bytes =
        std::uint64_t{h.mtp_position} * h.kv_row * sizeof(__half);
    if (!region(mtp != nullptr ? mtp->k_cache : nullptr, mtp_kv_bytes,
                "draft K cache") ||
        !region(mtp != nullptr ? mtp->v_cache : nullptr, mtp_kv_bytes,
                "draft V cache") ||
        !region(mtp != nullptr ? mtp->h : nullptr,
                std::uint64_t{h.hc_dim} * sizeof(float), "draft residual") ||
        !region(mtp != nullptr ? mtp->target_hidden : nullptr,
                std::uint64_t{h.hidden_rows} * h.hc_dim * sizeof(float),
                "kept trunk rows")) {
      return 0;
    }
  }
  return offset +
         std::uint64_t{h.image_count} * sizeof(qwen::vision::ImageGrid);
}

std::uint64_t Executor::SnapshotBytes(const Session& session,
                                      std::uint32_t hidden_rows) const {
  SnapshotHeader h =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  h.blocks = session.blocks_;
  h.mtp_blocks = session.mtp_.blocks;
  h.mtp_position = session.mtp_.position;
  h.hidden_rows = hidden_rows;
  return WalkSnapshot(
      h, nullptr,
      [](void*, std::uint64_t, std::uint64_t, const char*) { return true; });
}

bool Executor::SaveSnapshot(const Session& session, std::uint32_t hidden_rows,
                            std::span<std::uint8_t> payload,
                            std::string* error_msg) const {
  if (session.owner_ != this) {
    AssignError(error_msg, "session belongs to another executor");
    return false;
  }
  if (session.spec_tokens_ != 0) {
    AssignError(error_msg,
                "snapshot with a pending speculative batch; roll back first");
    return false;
  }
  if (hidden_rows > session.position_ ||
      hidden_rows > options_.max_speculative) {
    AssignError(error_msg, "kept trunk rows exceed the position or batch");
    return false;
  }
  SnapshotHeader h =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  h.blocks = session.blocks_;
  h.mtp_blocks = session.mtp_.blocks;
  h.mtp_position = session.mtp_.position;
  h.hidden_rows = hidden_rows;
  h.ngram_prev = session.ngram_.prev;
  h.payload_bytes = SnapshotBytes(session, hidden_rows);
  if (payload.size() != h.payload_bytes) {
    AssignError(error_msg, "snapshot buffer size does not match the payload");
    return false;
  }
  gufo::hip::SnapshotTransfer transfer;
  std::memcpy(payload.data(), &h, sizeof(h));
  const auto grid_bytes =
      std::size_t{h.image_count} * sizeof(qwen::vision::ImageGrid);
  if (grid_bytes != 0) {
    std::memcpy(payload.data() + payload.size() - grid_bytes,
                session.VisionLayout().images.data(), grid_bytes);
  }
  return WalkSnapshot(h, &session,
                      [&](void* device, std::uint64_t offset,
                          std::uint64_t bytes, const char*) {
                        if (!session.CheckCancellation(error_msg))
                          return false;
                        transfer.Copy(payload.data() + offset, device, bytes);
                        return true;
                      }) != 0;
}

bool Executor::RestoreSnapshot(Session& session,
                               std::span<const std::uint8_t> payload,
                               SnapshotInfo* info, std::string* error_msg,
                               std::uint32_t next_drafts) const {
  if (session.owner_ != this) {
    AssignError(error_msg, "session belongs to another executor");
    return false;
  }
  if (payload.size() < sizeof(SnapshotHeader)) {
    AssignError(error_msg, "snapshot payload is truncated");
    return false;
  }
  SnapshotHeader h{};
  std::memcpy(&h, payload.data(), sizeof(h));
  const SnapshotHeader mine =
      MakeSnapshotHeader(config(), session.mtp_enabled_, session);
  if (!SameGeometry(h, mine)) {
    AssignError(error_msg, "snapshot was taken with another model geometry");
    return false;
  }
  const std::uint32_t max_blocks =
      h.compress_ratio == 0 ? 0 : h.position / h.compress_ratio;
  if (h.image_count > 256 || h.position == 0 ||
      h.position > session.max_context_ || h.blocks > max_blocks ||
      h.mtp_position > h.position ||
      h.mtp_blocks > h.mtp_position / h.compress_ratio ||
      h.mtp_position - h.mtp_blocks * h.compress_ratio >
          session.index_capacity_ ||
      h.hidden_rows > h.position || h.hidden_rows > options_.max_speculative ||
      (h.has_mtp == 0 && (h.mtp_position != 0 || h.hidden_rows != 0))) {
    AssignError(error_msg, "snapshot positions do not fit this session");
    return false;
  }
  if (h.position - h.blocks * h.compress_ratio > session.index_capacity_) {
    AssignError(error_msg, "unpooled indexer rows exceed the ring capacity");
    return false;
  }
  if (h.payload_bytes != payload.size() ||
      WalkSnapshot(h, nullptr,
                   [](void*, std::uint64_t, std::uint64_t, const char*) {
                     return true;
                   }) != payload.size()) {
    AssignError(error_msg, "snapshot payload size does not match its header");
    return false;
  }
  qwen::vision::RopeLayout layout;
  layout.images.resize(h.image_count);
  const auto grid_bytes =
      layout.images.size() * sizeof(qwen::vision::ImageGrid);
  if (grid_bytes != 0) {
    std::memcpy(layout.images.data(),
                payload.data() + payload.size() - grid_bytes, grid_bytes);
  }
  try {
    layout.Validate(session.max_context_);
  } catch (const std::exception& e) {
    AssignError(error_msg, e.what());
    return false;
  }
  if (!layout.images.empty() && layout != session.VisionLayout()) {
    AssignError(error_msg,
                "image snapshot layout does not match its prompt attachment");
    return false;
  }
  if (layout.images.empty())
    session.ConfigureVision(nullptr, nullptr, stream_);
  // The session's queued work targets buffers the copies overwrite.
  if (!Check(hipStreamSynchronize(stream_), "restore drain", error_msg)) {
    return false;
  }
  // Every live state region is overwritten below. Retain only scratch that
  // the restored operation can use; this never allocates new rollback rows.
  // Explicit reset and failed restoration still release all scratch.
  const auto remaining = session.max_context_ - h.position;
  session.TrimRollback(std::min({next_drafts, options_.max_speculative - 1,
                                 remaining ? remaining - 1 : 0}));
  if (WalkSnapshot(h, &session,
                   [&](void* device, std::uint64_t offset, std::uint64_t bytes,
                       const char* what) {
                     return session.CheckCancellation(error_msg) &&
                            Check(hipMemcpy(device, payload.data() + offset,
                                            bytes, hipMemcpyHostToDevice),
                                  what, error_msg);
                   }) == 0) {
    session.Reset();
    return false;
  }
  session.RestoreVisionLayout(layout, stream_);
  session.position_ = h.position;
  session.blocks_ = h.blocks;
  session.mtp_.position = h.mtp_position;
  session.mtp_.blocks = h.mtp_blocks;
  session.spec_base_ = h.position;
  session.spec_tokens_ = 0;
  session.ngram_.prev = h.ngram_prev;
  if (info != nullptr) {
    info->position = h.position;
    info->hidden_rows = h.hidden_rows;
  }
  return true;
}

bool Executor::GreedyMtpPredictions(std::span<ArgmaxCandidate> predictions,
                                    std::string* error_msg) const {
  const auto rows = predictions.size();
  if (rows == 0 || rows > options_.max_logit_rows || rows > kArgmaxParts ||
      s_.mtp_ids == nullptr) {
    AssignError(error_msg, "invalid greedy MTP verification request");
    return false;
  }
  // Proposal selection has finished. Its two ID buffers and argmax scratch
  // can be reused until the next draft head overwrites them.
  const auto vocab = config().vocab_size;
  gufo::hip::LaunchBatchedGPUArgmax(
      VerificationLogits(), s_.mtp_ids, rows, vocab,
      {reinterpret_cast<float*>(s_.mtp_scratch_ids),
       MtpCandidateWorkspaceSize(vocab)},
      stream_);
  GatherArgmaxCandidates(VerificationLogits(), s_.mtp_ids, s_.mtp_argmax,
                         static_cast<std::uint32_t>(rows), vocab, stream_);
  return Check(hipMemcpyAsync(predictions.data(), s_.mtp_argmax,
                              predictions.size_bytes(), hipMemcpyDeviceToHost,
                              stream_),
               "greedy MTP predictions download", error_msg) &&
         Check(hipStreamSynchronize(stream_), "greedy MTP verification",
               error_msg);
}

bool Executor::MtpForward(Session& session,
                          std::span<const std::int32_t> tokens,
                          std::int32_t hidden_row, MtpOutput output,
                          std::string* error_msg,
                          const float* hidden_source) const {
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (session.owner_ != this ||
      std::any_of(tokens.begin(), tokens.end(), [&](auto t) {
        return t < 0 || static_cast<std::uint32_t>(t) >= config().vocab_size;
      })) {
    AssignError(error_msg, "invalid MTP session or token");
    return false;
  }
  if (!session.CheckCancellation(error_msg))
    return false;
  if (!session.mtp_enabled_) {
    AssignError(error_msg, "no MTP block loaded");
    return false;
  }

  if (n == 0 || n > options_.max_batch || (hidden_row < 0 && n != 1) ||
      (hidden_row >= 0 &&
       static_cast<std::uint32_t>(hidden_row) + n >
           (hidden_source ? options_.max_batch : options_.max_speculative))) {
    AssignError(error_msg, "MTP batch outside the kept hidden rows");
    return false;
  }
  if (output.trace != nullptr &&
      !output.trace->Valid(config().hidden_size, config().HcDim())) {
    AssignError(error_msg, "invalid MTP trace destinations");
    return false;
  }
  const std::uint32_t pos = session.mtp_.position;
  if (pos + n > session.max_context_) {
    AssignError(error_msg, "MTP context is full");
    return false;
  }
  ++session.mutation_epoch_;
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = session.position_;
  control_host_->blocks = session.blocks_;
  control_host_->mtp_position = pos;
  control_host_->hidden_row = hidden_row;
  control_host_->mtp_blocks = session.mtp_.blocks;
  const auto& c = config();
  const bool sparse = pos + n > c.indexer_top_k;
  const auto complete = (pos + n) / c.compress_ratio;
  const auto pool = sparse ? complete - session.mtp_.blocks : 0;
  const auto graph_pool = n / c.compress_ratio + 1;
  const bool graph = output.trace == nullptr && hidden_source == nullptr &&
                     n <= kVecBatch && pool <= graph_pool &&
                     pos + 1 >= session.VisionLayout().PrefixLength();
  const std::uint64_t key =
      static_cast<std::uint64_t>(n) |
      (static_cast<std::uint64_t>(hidden_row < 0) << 32) |
      (std::uint64_t{1} << 40) |
      (std::uint64_t{output.token != nullptr} << 41) |
      (std::uint64_t{output.candidates != nullptr} << 43) |
      (std::uint64_t{sparse} << 44);
  const auto body = [&] {
    return MtpBody(session, n, pos, output.token != nullptr,
                   output.candidates != nullptr, error_msg,
                   graph ? graph_pool : pool, hidden_source, output.trace);
  };
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  if (output.token != nullptr) {
    *output.token = *mtp_token_host_;
  }
  if (output.candidates != nullptr) {
    *output.candidates = *mtp_candidates_host_;
  }
  session.mtp_.position = pos + n;
  if (sparse)
    session.mtp_.blocks = complete;
  return true;
}

bool Executor::MtpBody(Session& session, std::uint32_t n, std::uint32_t pos,
                       bool token, bool candidates, std::string* error_msg,
                       std::uint32_t pool_grid, const float* hidden_source,
                       MtpTrace* trace) const {
  const Config& c = config();
  const DeviceLayer& l = model_->mtp();
  const std::uint32_t hc_dim = c.HcDim();
  const std::size_t final_row = static_cast<std::size_t>(n - 1) * hc_dim;
  const auto copy_trace = [&](const float* source,
                              std::span<float> destination) {
    return destination.empty() ||
           Check(hipMemcpyAsync(destination.data(), source,
                                destination.size_bytes(), hipMemcpyDeviceToHost,
                                stream_),
                 "MTP trace download", error_msg);
  };
  if (!Check(hipMemcpyAsync(session.control_, control_host_,
                            sizeof(Session::Control), hipMemcpyHostToDevice,
                            stream_),
             "control upload", error_msg) ||
      !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             "token upload", error_msg)) {
    return false;
  }
  EmbedTokens(model_->token_embd().data, SmallType(model_->token_embd().type),
              s_.tokens, s_.mtp_embd, n, c.hidden_size, 1, stream_);
  // MTP embeds shifted token IDs. Visual information arrives in the trunk
  // hidden stream; image embeddings belong only to the target input.
  RmsNormRows(s_.mtp_embd, l.nextn_enorm.f32(), s_.mtp_embd, n, c.hidden_size,
              1, c.rms_eps, stream_);
  // The hidden input: kept trunk rows from `hidden_row`, or the block's
  // own carried residual.
  MtpHidden(hidden_source ? hidden_source : session.mtp_.target_hidden,
            session.mtp_.h, &session.control_->hidden_row, s_.mtp_h, n, hc_dim,
            stream_);
  // Unlike the HC mixers, this normalization spans the complete HC * H row.
  RmsNormRows(s_.mtp_h, l.nextn_hnorm.f32(), s_.mtp_h, n, hc_dim, 1, c.rms_eps,
              stream_);
  if (trace && !copy_trace(s_.mtp_h + final_row, trace->normalized_hidden))
    return false;
  if (!Dense(l.nextn_fc_embedding, s_.mtp_embd, s_.mtp_eproj, n, error_msg) ||
      !Dense(l.nextn_fc_hidden, s_.mtp_h, s_.mtp_res, n * c.hc_count,
             error_msg)) {
    return false;
  }
  MtpAddEmbedding(s_.mtp_eproj, s_.mtp_res, n, c.hidden_size, c.hc_count,
                  stream_);
  if (trace && !copy_trace(s_.mtp_res + final_row, trace->fused))
    return false;
  Session::AttentionState attn;
  attn.rope = session.vision_input_.rope();
  attn.k_cache = session.mtp_.k_cache;
  attn.v_cache = session.mtp_.v_cache;
  attn.index_k = session.mtp_.index_k;
  attn.block_k = session.mtp_.block_k;
  // The draft block's attention runs at its own position.
  // Catch-up exports KV for every row but only carries its final residual.
  // Preserve that row's original query tile; earlier attention results
  // never contribute to the carried state.
  const bool last_only =
      !token && !candidates && n > 32 && control_host_->hidden_row >= 0;
  // Keep the final 128-column tile (and its predecessor for short tails).
  // At least 96 rows retain the wide mixer/projection arithmetic. Attention
  // still writes every KV/indexer row before the scratch view is narrowed.
  const auto skipped = last_only && n >= 224 ? (n - 96) / 128 * 128 : 0U;
  if (!HcMix(l.hc_attn, s_.mtp_res, false, s_.mixed, s_.inject, n, error_msg) ||
      !Attention(l, attn, s_.mixed, s_.block_out, n,
                 &session.control_->mtp_position, &session.control_->mtp_blocks,
                 pos, pool_grid, session.max_context_,
                 pos + n > c.indexer_top_k, error_msg, last_only, false,
                 skipped == 0)) {
    return false;
  }
  struct RestoreScratch {
    const Executor* executor;
    Scratch scratch;
    bool changed;
    ~RestoreScratch() {
      if (changed)
        executor->UseScratch(scratch);
    }
  } restore{this, s_, skipped != 0};
  const auto tail_rows = n - skipped;
  const auto tail_last = static_cast<std::size_t>(tail_rows - 1) * hc_dim;
  if (skipped != 0) {
    auto tail = RowScratch(s_, skipped);
    // Keep the producer's inject stride: it may use vectorized partials or
    // a single quantized projection rather than RowScratch's scalar layout.
    tail.inject = s_.inject + static_cast<std::size_t>(skipped) * c.hc_count *
                                  inject_parts_;
    UseScratch(tail);
    if (!Dense(l.attn_out, s_.ctx, s_.block_out, tail_rows, error_msg))
      return false;
  }
  Combine(s_.mtp_res, l.hc_ffn.norm.f32(), tail_rows);
  if (trace && !copy_trace(s_.mtp_res + tail_last, trace->attention))
    return false;
  if (!session.CheckCancellation(error_msg))
    return false;
  if (!HcMix(l.hc_ffn, s_.mtp_res, true, s_.mixed, s_.inject, tail_rows,
             error_msg) ||
      (trace && !copy_trace(s_.mixed + static_cast<std::size_t>(tail_rows - 1) *
                                           c.hidden_size,
                            trace->ffn_input)) ||
      !Moe(l, s_.mixed, s_.block_out, tail_rows, error_msg, last_only)) {
    return false;
  }
  Combine(s_.mtp_res, nullptr, tail_rows);
  if (trace &&
      !copy_trace(s_.block_out +
                      static_cast<std::size_t>(tail_rows - 1) * c.hidden_size,
                  trace->ffn_output))
    return false;
  if (trace && !copy_trace(s_.mtp_res + tail_last, trace->hidden))
    return false;
  if (trace && !trace->head.empty()) {
    if (!HcMix(l.nextn_head, s_.mtp_res + tail_last, false, s_.mixed, nullptr,
               1, error_msg) ||
        !copy_trace(s_.mixed, trace->head))
      return false;
  }
  const float* last = s_.mtp_res + tail_last;
  if (!Check(hipMemcpyAsync(session.mtp_.h, last, hc_dim * sizeof(float),
                            hipMemcpyDeviceToDevice, stream_),
             "MTP hidden carry", error_msg)) {
    return false;
  }
  return (!token && !candidates) ||
         MtpHead(l.nextn_head, last, token, candidates, error_msg);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
