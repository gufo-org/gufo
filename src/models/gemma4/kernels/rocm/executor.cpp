#include "src/models/gemma4/kernels/rocm/executor.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/managed_alloc.hpp"
#include "src/models/gemma4/continuation_hooks.hpp"
#include "src/models/gemma4/draft_vocab.hpp"
#include "src/models/gemma4/kernels/rocm/gemv.hpp"
#include "src/models/gemma4/kernels/rocm/kernels.hpp"
#include "src/models/gemma4/kernels/rocm/moe.hpp"
#include "src/models/gemma4/kernels/rocm/wmma_gemv.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

constexpr std::size_t kAlign = 256;
constexpr std::size_t kCandidateHeaderBytes = 256;

std::size_t AlignUp(std::size_t n) {
  return (n + kAlign - 1) / kAlign * kAlign;
}

struct Layout {
  std::size_t tokens, logit_index, key_ends, x, h, q, k, v, attn, o, gate, up,
      hsel, logits, partials, q8, act, wmma;
  std::size_t x_half, moe_x_half, moe_sync, moe_h, moe_logits, moe_ids,
      moe_weights, moe_groups, moe_gu, moe_act, moe_out, moe_counts, moe_bounds,
      moe_cursors, moe_rows_token, moe_rows_slot, moe_tiles;
  std::size_t draft_tokens, draft_embed, draft_concat, draft_x, draft_h,
      draft_q, draft_attn, draft_o, draft_gate, draft_up, draft_act,
      draft_logits, draft_next, draft_candidates, draft_candidate_scratch;
  std::size_t total;
};

std::uint32_t MaxQDim(const Config& c) {
  std::uint32_t m = 0;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    m = std::max(m, c.QDim(l));
  }
  return m;
}

std::uint32_t MaxKvDim(const Config& c) {
  std::uint32_t m = 0;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    m = std::max(m, c.KvDim(l));
  }
  return m;
}

/// Token rows per routed binary16 GEMM tile of Flash-Next's kernel: 48 for a
/// Q8_0 gate/up, 64 for the Q5_1/Q8_0 down projection (2.5% faster than 48).
constexpr std::uint32_t kRoutedTileRows = 48;
constexpr std::uint32_t kRoutedDownRows = 64;
/// The Gemma K-quant gate/up GEMM skips a tile's empty 16-row parts, so
/// wide tiles cost no padding work: Q6_K 96 rows 4.3 ms per 2048-row layer,
/// 48 rows 6.3, 128 rows 4.7 (one block per WGP).
constexpr std::uint32_t kRoutedKQuantTileRows = kRoutedHalfTileRows;

/// Whether layer l's experts take the binary16 prefill route (the formats
/// PrefillExperts covers).
bool HalfExperts(const DeviceLayer& l, const Config& c);

Layout Plan(const Config& c, const Config* draft, std::uint32_t vocab,
            std::size_t max_cols, std::size_t max_half_cols, std::uint32_t rows,
            std::uint32_t logit_rows, std::uint32_t max_context) {
  const std::size_t f = sizeof(float);
  const std::uint32_t ring = RingSlots(c, rows);
  const std::size_t partial_floats = std::max(
      AttentionPartialFloats(rows, c.num_heads, c.head_dim_global, max_context),
      AttentionPartialFloats(rows, c.num_heads, c.head_dim_sliding,
                             std::min(max_context, ring)));
  Layout l{};
  std::size_t at = 0;
  const auto take = [&](std::size_t bytes) {
    const std::size_t here = at;
    at += AlignUp(bytes);
    return here;
  };
  l.tokens = take(rows * sizeof(std::int32_t));
  l.logit_index = take(logit_rows * sizeof(std::uint32_t));
  l.key_ends = take(rows * sizeof(std::uint32_t));
  l.x = take(std::size_t{rows} * c.hidden_size * f);
  l.h = take(std::size_t{rows} * c.hidden_size * f);
  l.q = take(std::size_t{rows} * MaxQDim(c) * f);
  // K and V rows, or the [q | k | v] rows of a fused projection.
  l.k = take(std::size_t{rows} * (MaxQDim(c) + 2 * MaxKvDim(c)) * f);
  l.v = l.k + std::size_t{rows} * MaxKvDim(c) * f;
  l.attn = take(std::size_t{rows} * MaxQDim(c) * f);
  l.o = take(std::size_t{rows} * c.hidden_size * f);
  // Gate then up rows, or the [gate | up] rows of a fused projection.
  l.gate = take(std::size_t{rows} * 2 * c.ffn_size * f);
  l.up = l.gate + std::size_t{rows} * c.ffn_size * f;
  l.act = take(std::size_t{rows} * c.ffn_size * f);
  l.hsel = take(std::size_t{logit_rows} * c.hidden_size * f);
  l.logits = take(std::size_t{logit_rows} * vocab * f);
  l.partials = take(partial_floats * f);
  l.wmma = take(WmmaGemvScratchBytes(static_cast<std::uint32_t>(max_cols)));
  const std::uint32_t q8_rows = std::max(rows, logit_rows);
  l.q8 = take(q8_rows > kSplitRows
                  ? hip::QuantizedActivationBytes(q8_rows, max_cols)
                  : 0);
  // Binary16 activation rows: every prefill input of an expert model, the
  // inputs of binary16 projections otherwise (all of a binary16-prefill
  // dense model's, whose max_half_cols is its widest projection).
  const std::size_t half_cols = c.HasExperts() ? max_cols : max_half_cols;
  l.x_half =
      take(q8_rows > kSplitRows ? std::size_t{q8_rows} * half_cols * 2 : 0);
  if (c.HasExperts()) {
    // Prefill stages binary16 activations; the expert buffers hold FP32 rows
    // at grouped widths and binary16 rows in prefill.
    const std::size_t slots = std::size_t{rows} * c.experts_used;
    const std::size_t grouped =
        std::size_t{std::min(rows, kMaxGroupSlots)} * c.experts_used;
    const std::size_t width = c.expert_ffn_size;
    l.moe_x_half = take(std::size_t{rows} * c.hidden_size * 2);
    l.moe_h = take(std::size_t{rows} * c.hidden_size * f);
    l.moe_logits = take(std::size_t{rows} * c.num_experts * f);
    l.moe_ids = take(slots * sizeof(std::int32_t));
    l.moe_weights = take(slots * f);
    l.moe_groups = take(ExpertGroupInts(std::min<std::uint32_t>(
                            c.num_experts, kMaxGroupSlots * c.experts_used)) *
                        sizeof(std::int32_t));
    l.moe_gu = take(std::max(slots * 2 * width * 2, grouped * 2 * width * f));
    l.moe_act = take(std::max(slots * width * 2, grouped * width * f));
    l.moe_out = take(slots * c.hidden_size * f);
    l.moe_counts = take(c.num_experts * sizeof(std::uint32_t));
    l.moe_sync = take(sizeof(std::uint32_t));
    l.moe_bounds = take((c.num_experts + 1) * sizeof(std::int32_t));
    l.moe_cursors = take(c.num_experts * sizeof(std::int32_t));
    const std::size_t compact =
        qwen38_flash_next::rocm::RoutedCompactRows(slots, c.num_experts);
    l.moe_rows_token = take(compact * sizeof(std::int32_t));
    l.moe_rows_slot = take(compact * sizeof(std::int32_t));
    // The gate/up map (the shorter tiles at most) and the down map.
    const auto map_slots = static_cast<std::uint32_t>(slots);
    l.moe_tiles =
        take((RoutedTileCapacity(map_slots, c.num_experts, kRoutedTileRows) +
              RoutedTileCapacity(map_slots, c.num_experts, kRoutedDownRows)) *
             sizeof(std::int32_t));
  }
  if (draft != nullptr) {
    // One row per session drafting in the same step.
    const std::size_t n = kMaxDraftSessions;
    l.draft_tokens = take(n * sizeof(std::uint32_t));
    l.draft_embed = take(n * draft->target_hidden_size * f);
    l.draft_concat = take(n * 2 * draft->target_hidden_size * f);
    l.draft_x = take(n * draft->hidden_size * f);
    l.draft_h = take(n * draft->hidden_size * f);
    l.draft_q = take(n * MaxQDim(*draft) * f);
    l.draft_attn = take(n * MaxQDim(*draft) * f);
    l.draft_o = take(n * draft->hidden_size * f);
    l.draft_gate = take(n * 2 * draft->ffn_size * f);
    l.draft_up = l.draft_gate + n * draft->ffn_size * f;
    l.draft_act = take(n * draft->ffn_size * f);
    l.draft_logits = take(n * vocab * f);
    l.draft_next = take(n * draft->target_hidden_size * f);
    // Top-64 proposal candidates for sampled drafting: ids, then scores.
    const std::size_t candidate_ids =
        qwen38_flash_next::rocm::MtpCandidateWorkspaceSize(vocab);
    l.draft_candidates = take(n * candidate_ids * sizeof(std::uint32_t));
    l.draft_candidate_scratch = take(n * candidate_ids * sizeof(std::uint32_t));
  }
  l.total = at;
  return l;
}

}  // namespace

KvCache::~KvCache() {
  if (allocation != nullptr) {
    hip::LogCleanupError(hipFree(allocation));
  }
}

std::uint32_t RingSlots(const Config& config, std::uint32_t max_rows) noexcept {
  const std::uint32_t slots = config.sliding_window + max_rows;
  return (slots + 255U) / 256U * 256U;
}

std::size_t Executor::ScratchBytes(const Config& config, const Config* draft,
                                   std::uint32_t vocab, std::size_t max_cols,
                                   std::size_t max_half_cols,
                                   std::uint32_t max_rows,
                                   std::uint32_t max_logit_rows,
                                   std::uint32_t max_context) {
  return Plan(config, draft, vocab, max_cols, max_half_cols, max_rows,
              max_logit_rows, max_context)
      .total;
}

std::size_t Executor::CacheBytes(const Config& c, std::uint32_t max_context,
                                 std::uint32_t ring,
                                 const std::vector<std::uint32_t>& key_widths,
                                 std::uint32_t vocab,
                                 std::size_t draft_table_bytes) {
  std::size_t bytes = 0;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::size_t slots =
        c.IsSliding(l) ? std::min(ring, max_context) : max_context;
    bytes += AlignUp(slots * key_widths[l] * sizeof(std::uint16_t)) +
             AlignUp(slots * c.KvDim(l) * sizeof(std::uint16_t));
  }
  return bytes + AlignUp(std::size_t{c.hidden_size} * sizeof(float)) +
         AlignUp(std::size_t{kSplitRows} * c.hidden_size * sizeof(float)) +
         AlignUp(std::size_t{vocab} * sizeof(float)) +
         AlignUp(draft_table_bytes);
}

std::size_t Executor::draft_table_bytes() const noexcept {
  return model_.draft_head_tokens().empty()
             ? 0
             : std::size_t{kDraftTableRows} * model_.draft_row_bytes();
}

std::vector<std::uint32_t> Executor::KeyWidths(const DeviceModel& model) {
  const Config& c = model.config();
  std::vector<std::uint32_t> widths(c.num_layers);
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    // One projection feeds K and V: the cache keeps only the rotated key
    // dims; the others are the values scaled by k_norm.
    const std::uint32_t pairs = model.global_rope_pairs();
    const bool derived = !c.IsSliding(l) && c.HeadDim(l) == 512 &&
                         model.layers()[l].attn_v.empty() && pairs % 16 == 0 &&
                         pairs <= kMaxDerivedKeyPairs;
    widths[l] = derived ? c.kv_heads[l] * 2 * pairs : c.KvDim(l);
  }
  return widths;
}

Executor::Executor(const DeviceModel& model, std::uint32_t max_rows,
                   std::uint32_t max_logit_rows, std::uint32_t max_context)
    : model_(model),
      max_rows_(max_rows),
      max_logit_rows_(max_logit_rows),
      max_context_(max_context),
      ring_(RingSlots(model.config(), max_rows)),
      key_widths_(KeyWidths(model)) {
  const Layout l =
      Plan(model.config(), model.has_draft() ? &model.draft().config : nullptr,
           model.vocab_size(), model.max_cols(), model.max_half_cols(),
           max_rows, max_logit_rows, max_context);
  if (const Config& c = model.config();
      c.HasExperts() &&
      (c.hidden_size % 128 != 0 || c.hidden_size > kMaxRouterHidden ||
       c.num_experts % 8 != 0)) {
    throw std::invalid_argument(
        "gemma4 expert routing needs a hidden width that is a multiple of 128 "
        "up to " +
        std::to_string(kMaxRouterHidden) + " and experts in groups of 8");
  }
  HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  if (model.config().HasExperts()) {
    HIP_CHECK(hipStreamCreateWithFlags(&expert_stream_, hipStreamNonBlocking));
    HIP_CHECK(hipEventCreateWithFlags(&expert_fork_, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&expert_join_, hipEventDisableTiming));
    if (hipblasCreate(&blas_) != HIPBLAS_STATUS_SUCCESS ||
        hipblasSetStream(blas_, stream_) != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("hipblasCreate failed");
    }
  }
  HIP_CHECK(hip::AllocateDevice(&scratch_, l.total));
  auto* base = static_cast<std::uint8_t*>(scratch_);
  tokens_ = reinterpret_cast<std::int32_t*>(base + l.tokens);
  logit_index_ = reinterpret_cast<std::uint32_t*>(base + l.logit_index);
  key_ends_ = reinterpret_cast<std::uint32_t*>(base + l.key_ends);
  x_ = reinterpret_cast<float*>(base + l.x);
  h_ = reinterpret_cast<float*>(base + l.h);
  q_ = reinterpret_cast<float*>(base + l.q);
  k_ = reinterpret_cast<float*>(base + l.k);
  v_ = reinterpret_cast<float*>(base + l.v);
  attn_ = reinterpret_cast<float*>(base + l.attn);
  o_ = reinterpret_cast<float*>(base + l.o);
  gate_ = reinterpret_cast<float*>(base + l.gate);
  up_ = reinterpret_cast<float*>(base + l.up);
  act_ = reinterpret_cast<float*>(base + l.act);
  hsel_ = reinterpret_cast<float*>(base + l.hsel);
  logits_ = reinterpret_cast<float*>(base + l.logits);
  partials_ = reinterpret_cast<float*>(base + l.partials);
  q8_ = base + l.q8;
  wmma_ = base + l.wmma;
  x_half_ = base + l.x_half;
  half_prefill_ = model.half_prefill();
  if (model.config().HasExperts()) {
    const Config& c = model.config();
    moe_x_half_ = base + l.moe_x_half;
    moe_h_ = reinterpret_cast<float*>(base + l.moe_h);
    moe_logits_ = reinterpret_cast<float*>(base + l.moe_logits);
    moe_ids_ = reinterpret_cast<std::int32_t*>(base + l.moe_ids);
    moe_weights_ = reinterpret_cast<float*>(base + l.moe_weights);
    moe_groups_ = reinterpret_cast<std::int32_t*>(base + l.moe_groups);
    moe_gu_ = base + l.moe_gu;
    moe_act_ = base + l.moe_act;
    moe_out_ = reinterpret_cast<float*>(base + l.moe_out);
    moe_counts_ = reinterpret_cast<std::uint32_t*>(base + l.moe_counts);
    moe_sync_ = reinterpret_cast<std::uint32_t*>(base + l.moe_sync);
    HIP_CHECK(hipMemset(moe_sync_, 0, sizeof(std::uint32_t)));
    moe_bounds_ = reinterpret_cast<std::int32_t*>(base + l.moe_bounds);
    moe_cursors_ = reinterpret_cast<std::int32_t*>(base + l.moe_cursors);
    moe_rows_token_ = reinterpret_cast<std::int32_t*>(base + l.moe_rows_token);
    moe_rows_slot_ = reinterpret_cast<std::int32_t*>(base + l.moe_rows_slot);
    moe_tiles_ = reinterpret_cast<std::int32_t*>(base + l.moe_tiles);
  }
  if (model.has_draft()) {
    draft_tokens_ = reinterpret_cast<std::uint32_t*>(base + l.draft_tokens);
    draft_embed_ = reinterpret_cast<float*>(base + l.draft_embed);
    draft_concat_ = reinterpret_cast<float*>(base + l.draft_concat);
    draft_x_ = reinterpret_cast<float*>(base + l.draft_x);
    draft_h_ = reinterpret_cast<float*>(base + l.draft_h);
    draft_q_ = reinterpret_cast<float*>(base + l.draft_q);
    draft_attn_ = reinterpret_cast<float*>(base + l.draft_attn);
    draft_o_ = reinterpret_cast<float*>(base + l.draft_o);
    draft_gate_ = reinterpret_cast<float*>(base + l.draft_gate);
    draft_up_ = reinterpret_cast<float*>(base + l.draft_up);
    draft_act_ = reinterpret_cast<float*>(base + l.draft_act);
    draft_logits_ = reinterpret_cast<float*>(base + l.draft_logits);
    draft_next_ = reinterpret_cast<float*>(base + l.draft_next);
    draft_candidates_ =
        reinterpret_cast<std::uint32_t*>(base + l.draft_candidates);
    draft_candidate_scratch_ =
        reinterpret_cast<std::uint32_t*>(base + l.draft_candidate_scratch);
    HIP_CHECK(hipHostMalloc(
        &draft_candidates_host_,
        kMaxDraftSessions * sizeof(qwen38_flash_next::MtpCandidateLogits)));
  }
  HIP_CHECK(hipHostMalloc(&argmax_host_, kSplitRows * sizeof(std::uint32_t)));
  logits_host_.resize(std::size_t{max_logit_rows} * model.vocab_size());
  if (!logits_host_.empty()) {
    HIP_CHECK(hipHostRegister(logits_host_.data(),
                              logits_host_.size() * sizeof(float),
                              hipHostRegisterDefault));
  }
  if (model.vocab_size() % 4 == 0) {
    const std::size_t work = CandidateWorkBytes(kSplitRows);
    const std::size_t out =
        kCandidateHeaderBytes + std::size_t{kSplitRows} *
                                    (kCandidateTop + 2 * kMaxCandidateExtra) *
                                    sizeof(sampling::LogitCandidate);
    const std::size_t extra = kMaxCandidateExtra * sizeof(std::uint32_t);
    HIP_CHECK(hip::AllocateDevice(&candidate_scratch_, work + out + extra));
    auto* base = static_cast<std::uint8_t*>(candidate_scratch_);
    candidate_work_ = CandidateWorkAt(base, kSplitRows, stream_);
    candidate_out_ = base + work;
    candidate_extra_ = reinterpret_cast<std::uint32_t*>(base + work + out);
    candidate_host_.resize(out);
    candidate_extra_host_.resize(kMaxCandidateExtra);
    HIP_CHECK(hipHostRegister(candidate_host_.data(), candidate_host_.size(),
                              hipHostRegisterDefault));
    HIP_CHECK(hipHostRegister(candidate_extra_host_.data(),
                              kMaxCandidateExtra * sizeof(std::uint32_t),
                              hipHostRegisterDefault));
  }
}

Executor::~Executor() {
  if (!logits_host_.empty()) {
    hip::LogCleanupError(hipHostUnregister(logits_host_.data()));
  }
  if (!candidate_host_.empty()) {
    hip::LogCleanupError(hipHostUnregister(candidate_host_.data()));
    hip::LogCleanupError(hipHostUnregister(candidate_extra_host_.data()));
  }
  if (candidate_scratch_ != nullptr) {
    hip::LogCleanupError(hipFree(candidate_scratch_));
  }
  if (argmax_host_ != nullptr) {
    hip::LogCleanupError(hipHostFree(argmax_host_));
  }
  if (copy_runs_ != nullptr) {
    hip::LogCleanupError(hipFree(copy_runs_));
  }
  if (draft_candidates_host_ != nullptr) {
    hip::LogCleanupError(hipHostFree(draft_candidates_host_));
  }
  if (scratch_ != nullptr) {
    hip::LogCleanupError(hipFree(scratch_));
  }
  if (blas_ != nullptr) {
    (void)hipblasDestroy(blas_);
  }
  if (expert_fork_ != nullptr) {
    hip::LogCleanupError(hipEventDestroy(expert_fork_));
  }
  if (expert_join_ != nullptr) {
    hip::LogCleanupError(hipEventDestroy(expert_join_));
  }
  if (expert_stream_ != nullptr) {
    hip::LogCleanupError(hipStreamDestroy(expert_stream_));
  }
  if (stream_ != nullptr) {
    hip::LogCleanupError(hipStreamDestroy(stream_));
  }
}

std::unique_ptr<KvCache> Executor::CreateCache(std::uint32_t max_context,
                                               std::string* error_msg) const {
  const Config& c = model_.config();
  if (max_context == 0 || max_context > max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "session context exceeds the executor capacity";
    }
    return nullptr;
  }
  auto cache = std::make_unique<KvCache>();
  cache->max_context = max_context;
  cache->ring = ring_;
  cache->bytes = CacheBytes(c, max_context, ring_, key_widths_,
                            model_.vocab_size(), draft_table_bytes());
  if (hip::AllocateDevice(&cache->allocation, cache->bytes) != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg = "KV cache allocation failed (" +
                   std::to_string(cache->bytes) + " bytes)";
    }
    cache->allocation = nullptr;
    return nullptr;
  }
  auto* at = static_cast<std::uint8_t*>(cache->allocation);
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::size_t slots =
        c.IsSliding(l) ? std::min(ring_, max_context) : max_context;
    const std::size_t bytes =
        AlignUp(slots * c.KvDim(l) * sizeof(std::uint16_t));
    cache->k.push_back(reinterpret_cast<std::uint16_t*>(at));
    at += AlignUp(slots * key_widths_[l] * sizeof(std::uint16_t));
    cache->v.push_back(reinterpret_cast<std::uint16_t*>(at));
    at += bytes;
  }
  const std::size_t hidden_bytes =
      AlignUp(std::size_t{c.hidden_size} * sizeof(float));
  cache->hidden = reinterpret_cast<float*>(at);
  at += hidden_bytes;
  cache->ahead_hidden = reinterpret_cast<float*>(at);
  at += AlignUp(std::size_t{kSplitRows} * c.hidden_size * sizeof(float));
  cache->logits = reinterpret_cast<float*>(at);
  at += AlignUp(std::size_t{model_.vocab_size()} * sizeof(float));
  if (draft_table_bytes() != 0) {
    cache->draft_table = at;
  }
  return cache;
}

const void* Executor::Quantize(const float* x, std::uint32_t rows,
                               std::uint32_t cols) {
  if (rows <= kSplitRows) {
    return nullptr;
  }
  if (half_prefill_) {
    qwen38_flash_next::rocm::NarrowActivations(
        x, x_half_, false, std::size_t{rows} * cols, stream_);
    return x_half_;
  }
  hip::LaunchQuantizeActivationQ8_1FromFp32(x, q8_, rows, cols, stream_);
  return q8_;
}

namespace {

std::optional<GemvFormat> GemvFormatOf(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ4_0:
      return GemvFormat::kQ4_0;
    case core::GgmlType::kQ4_K:
      return GemvFormat::kQ4_K;
    case core::GgmlType::kQ5_K:
      return GemvFormat::kQ5_K;
    case core::GgmlType::kQ6_K:
      return GemvFormat::kQ6_K;
    case core::GgmlType::kQ8_0:
      return GemvFormat::kQ8_0;
    default:
      return std::nullopt;
  }
}

/// The binary16 GEMM plan for a Q8_0 projection of a Gemma width (the
/// Flash-Next dispatch is tuned on its own shapes): eight row groups for the
/// 26B-A4B's dense MLP down (K = 2112), 605 -> 582 us per 2048 rows. The
/// 31B's 5376 outputs (21 row tiles) take two-block stages and walk seven
/// row tiles of a token tile in launch order: per pp2048 chunk, attention
/// output 473 -> 344 ms and MLP down 824 -> 769 ms. Its other shapes
/// measured slower with every staged or walking plan.
qwen38_flash_next::rocm::DenseF16Plan HalfPlan(std::uint32_t m,
                                               std::uint32_t k) {
  using qwen38_flash_next::rocm::DenseF16Plan;
  if (m == 5376 && k >= 8192) {
    return DenseF16Plan::kStagedRowGroups8Walk7;
  }
  return m == 2816 && k == 2112 ? DenseF16Plan::kRowGroups8
                                : DenseF16Plan::kAuto;
}

/// The formats of the dense binary16 prefill GEMM (LaunchDenseHalfGemm).
std::optional<ExpertFormat> DenseHalfFormat(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ4_0:
      return ExpertFormat::kQ4_0;
    case core::GgmlType::kQ5_0:
      return ExpertFormat::kQ5_0;
    case core::GgmlType::kQ4_K:
      return ExpertFormat::kQ4_K;
    case core::GgmlType::kQ5_K:
      return ExpertFormat::kQ5_K;
    case core::GgmlType::kQ6_K:
      return ExpertFormat::kQ6_K;
    default:
      return std::nullopt;
  }
}

/// Whether a prefill of `rows` reads w's prefill copy: binary16 weights
/// only pay from 512 rows, where the weight reads stop dominating.
bool PackedPrefill(const DeviceTensor& w, std::uint32_t rows) {
  switch (w.prefill_layout) {
    case PrefillLayout::kCompact:
      return true;
    case PrefillLayout::kDecoded:
      return rows >= 512;
    default:
      return false;
  }
}

/// Decode and verification projections on the routed GEMV's dense mode
/// (LaunchDenseGemv): Q5_0, which no other GEMV decodes, and Q4_0 rows the
/// Gemma GEMVs cannot split into 256-value groups (the 26B-A4B's 2112-wide
/// down projection).
std::optional<ExpertFormat> DenseGemvFormat(core::GgmlType type,
                                            std::uint32_t k) {
  if (type == core::GgmlType::kQ5_0) {
    return ExpertFormat::kQ5_0;
  }
  if (type == core::GgmlType::kQ4_0 && k % 256 != 0) {
    return ExpertFormat::kQ4_0;
  }
  return std::nullopt;
}

}  // namespace

bool Executor::WmmaProjection(const DeviceTensor& w, std::uint32_t rows) const {
  return w.type != core::GgmlType::kF16 && rows >= 1 && rows <= kSplitRows &&
         rows <= kWmmaGemvRows && !DenseGemvFormat(w.type, w.cols) &&
         w.cols >= 2048 && (model_.has_draft() || rows >= 6) &&
         WmmaGemvSupports(w.type, w.rows, w.cols);
}

WmmaPackOut Executor::PackFor(const DeviceTensor& w, std::uint32_t rows) const {
  return WmmaProjection(w, rows) ? WmmaPackOf(wmma_, w.cols) : WmmaPackOut{};
}

void Executor::Project(const DeviceTensor& w, const float* x, const void* xq,
                       std::uint32_t rows, float* y, bool packed) {
  if (w.type == core::GgmlType::kF16) {
    ProjectHalf(w, x, xq, rows, y);
    return;
  }
  if (rows > kSplitRows) {
    if (half_prefill_) {
      if (PackedPrefill(w, rows) &&
          LaunchPackedPrefillGemm(w.prefill_format, w.prefill_layout, w.prefill,
                                  xq, y, nullptr, rows, w.rows, w.cols,
                                  stream_)) {
        return;
      }
      // Binary16 activations: Q8_0, Q4_0 and K-quant weights decode to
      // binary16 in the WMMA GEMMs; other formats keep the W8A8 route on their
      // own Q8_1 rows.
      // The 26B-A4B's Q8_0 attention outputs (2816 outputs, 11 row tiles
      // of Flash-Next's 256-row blocks) fill the WGPs better on Gemma's
      // 128-row blocks, with the same sums: 1.80 -> 1.61 ms per 2048 rows.
      if (w.type == core::GgmlType::kQ8_0 && w.rows == 2816 && w.cols >= 4096 &&
          LaunchDenseHalfGemm(ExpertFormat::kQ8_0, w.data, xq, y, nullptr, rows,
                              w.rows, w.cols, stream_, false, w.grouped)) {
        return;
      }
      if (w.type == core::GgmlType::kQ8_0 &&
          qwen38_flash_next::rocm::DenseF16Gemm(
              w.data, static_cast<const __half*>(xq), y, rows, w.rows, w.cols,
              stream_, 0, HalfPlan(w.rows, w.cols), w.grouped)) {
        return;
      }
      if (const auto format = DenseHalfFormat(w.type);
          format &&
          LaunchDenseHalfGemm(*format, w.data, xq, y, nullptr, rows, w.rows,
                              w.cols, stream_, false, w.grouped)) {
        return;
      }
      hip::LaunchQuantizeActivationQ8_1FromFp32(x, q8_, rows, w.cols, stream_);
      xq = q8_;
    }
    if (w.grouped) {
      throw std::runtime_error(
          "gemma4 grouped weights reached a row-order GEMM");
    }
    hip::LaunchBatchedQuantGEMMPreQuantized(w.type, w.data, xq, y, rows, w.rows,
                                            w.cols, stream_);
    return;
  }
  // One kernel at every decode width, so verification rows round like
  // single tokens.
  if (const auto format = DenseGemvFormat(w.type, w.cols);
      format &&
      LaunchDenseGemv(*format, w.data, x, y, rows, w.rows, w.cols, stream_)) {
    return;
  }
  // Target projections take the WMMA kernel at every width when the drafter
  // is loaded (verification rows round like single tokens), and batched
  // autoregressive rows from six on, where it is faster.
  if (WmmaProjection(w, rows) &&
      LaunchWmmaGemv(w.type, w.data, x, y, rows, w.rows, w.cols, wmma_, stream_,
                     packed, w.grouped)) {
    return;
  }
  // DeviceModel::Upload groups only weights the WMMA projections take.
  if (w.grouped) {
    throw std::runtime_error("gemma4 grouped weights reached a row-order GEMV");
  }
  if (rows > 1) {
    // Gemma's K-quant shapes lie outside the Qwen-tuned dispatch; the
    // double-stage configuration is faster on them (the vocabulary head keeps
    // the default). Q4_0 keeps the default up to eight rows and takes the
    // double stage past that, where the default splits the rows into two
    // passes. All of them round like the one-row twins below.
    const bool double_stage =
        w.type == core::GgmlType::kQ4_0 ? rows > 8 : w.rows < 65536;
    if (double_stage &&
        hip::LaunchKQuantSmallBatchDoubleStage(w.type, w.data, x, y, rows,
                                               w.rows, w.cols, stream_)) {
      return;
    }
    // Q8_0 past eight rows reads its weights once instead of per eight rows.
    if (w.type == core::GgmlType::kQ8_0 &&
        hip::LaunchQ8_0SmallBatchWide(w.data, x, y, rows, w.rows, w.cols,
                                      stream_)) {
      return;
    }
    hip::LaunchBatchedQuantGEMMFp32(w.type, w.data, x, y, rows, w.rows, w.cols,
                                    stream_);
    return;
  }
  if (model_.has_draft()) {
    // Speculation verifies with the small-batch kernel, and single tokens
    // must round identically: its bit-identical one-row twins, whichever is
    // faster for the shape (measured with cold weights): the double-stage
    // pass for Q4_K and for Q5_K past 4096 outputs, the small-batch twin for
    // Q4_0 at every shape.
    const bool double_stage =
        w.type == core::GgmlType::kQ4_K ||
        (w.type == core::GgmlType::kQ5_K && w.rows > 4096);
    if (double_stage && hip::LaunchKQuantSmallBatchDoubleStage(
                            w.type, w.data, x, y, 1, w.rows, w.cols, stream_)) {
      return;
    }
    if (w.rows <= 4096 && w.type != core::GgmlType::kQ4_0) {
      hip::LaunchGEMV(w.data, w.type, x, y, w.rows, w.cols, stream_);
    } else {
      hip::LaunchBatchedQuantGEMMFp32(w.type, w.data, x, y, 1, w.rows, w.cols,
                                      stream_);
    }
    return;
  }
  // Autoregressive decode: the Gemma GEMV serves every K-quant and Q4_0
  // projection, and Q8_0 past 2112 outputs and 1024 inputs (cold weights:
  // 8192x5376 208 vs 217 us, 5376x16384 407 vs 506, 2816x8192 117 vs 123;
  // the shared GEMV is faster on the drafter's 1024-wide rows and on short
  // outputs such as 1024x2816, 18 vs 16 us).
  const bool gemma_gemv =
      w.type != core::GgmlType::kQ8_0 || (w.rows > 2112 && w.cols > 1024);
  if (const auto format = GemvFormatOf(w.type);
      format && gemma_gemv &&
      LaunchKQuantGemv(*format, w.data, x, y, w.rows, w.cols, stream_)) {
    return;
  }
  hip::LaunchGEMV(w.data, w.type, x, y, w.rows, w.cols, stream_);
}

void Executor::ProjectDraft(const DeviceTensor& w, const float* x,
                            std::uint32_t rows, float* y) {
  // Drafter projections have no decode twin to match (as the head): a lone
  // chain reads the shapes the WMMA kernel would take with the Gemma GEMV,
  // which spreads the drafter's 1024 outputs over every WGP. Cold weights,
  // Q4_0: 1024 x 10752 33.7 us (shared small-batch 40.3), 1024 x 8192 27.1
  // (35.2).
  if (rows == 1 && w.cols >= 2048 && WmmaGemvSupports(w.type, w.rows, w.cols)) {
    if (const auto format = GemvFormatOf(w.type);
        format &&
        LaunchKQuantGemv(*format, w.data, x, y, w.rows, w.cols, stream_)) {
      return;
    }
  }
  Project(w, x, nullptr, rows, y);
}

static_assert(
    kMaxHalfGemvRows == kSplitRows,
    "the binary16 GEMV serves every row count below the prefill GEMM");

void Executor::ProjectHalf(const DeviceTensor& w, const float* x,
                           const void* xq, std::uint32_t rows, float* y) {
  if (rows <= kSplitRows) {
    // One kernel at every decode width: verification rows round like
    // single tokens.
    if (!LaunchHalfGemv(w.data, x, y, rows, w.rows, w.cols, stream_)) {
      throw std::runtime_error("gemma4 binary16 projection shape unsupported");
    }
    return;
  }
  // Binary16 prefill inputs come from their producers; the W8A8 prefill
  // narrows the FP32 rows here.
  const void* xh = xq;
  if (!half_prefill_) {
    qwen38_flash_next::rocm::NarrowActivations(
        x, x_half_, false, std::size_t{rows} * w.cols, stream_);
    xh = x_half_;
  }
  // Every Gemma projection is thousands of outputs wide: 128 x 128 tiles
  // from 128 rows (31B Q8 binary16 layers at 512 rows: q 1.91 -> 1.23 ms,
  // down 5.03 -> 3.46 ms).
  if (!qwen38_flash_next::rocm::UnquantizedF16Gemm(
          w.data, static_cast<const __half*>(xh), y, rows, w.rows, w.cols,
          stream_, /*wide_from=*/128)) {
    throw std::runtime_error("gemma4 binary16 prefill shape unsupported");
  }
}

bool Executor::DerivedKeys(std::uint32_t layer) const {
  return key_widths_[layer] != model_.config().KvDim(layer);
}

void Executor::DeriveKeys(AttentionArgs& att, std::uint32_t layer) const {
  if (DerivedKeys(layer)) {
    att.rope_pairs = model_.global_rope_pairs();
  }
}

void Executor::Forward(KvCache& cache, std::span<const std::int32_t> tokens,
                       std::uint32_t first_position,
                       std::span<const std::uint32_t> logit_rows,
                       std::span<const ImageRows> images,
                       std::uint32_t siblings) {
  const Segment segment{&cache, first_position,
                        static_cast<std::uint32_t>(tokens.size()), siblings};
  Forward(std::span<const Segment>(&segment, 1), tokens, logit_rows, images);
}

void Executor::Forward(std::span<const Segment> segments,
                       std::span<const std::int32_t> tokens,
                       std::span<const std::uint32_t> logit_rows,
                       std::span<const ImageRows> images) {
  const Config& c = model_.config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  std::uint64_t total = 0;
  for (const Segment& s : segments) {
    if (s.rows == 0 || s.cache->ring != ring_ ||
        s.first_position + std::uint64_t{s.rows} > s.cache->max_context) {
      throw std::invalid_argument("gemma4 forward segment exceeds its cache");
    }
    // Siblings stand beside drafts: at most one per draft.
    if (s.siblings != 0 &&
        (2 * s.siblings >= s.rows || s.rows > kSplitRows || !images.empty())) {
      throw std::invalid_argument("gemma4 forward siblings exceed the chain");
    }
    total += s.rows;
  }
  for (const Segment& s : segments) {
    s.cache->written_from = std::min(s.cache->written_from, s.first_position);
    s.cache->written_to =
        std::max(s.cache->written_to, s.first_position + s.rows);
  }
  // Several sessions share a forward only at decode widths, where every
  // projection keeps its single-session arithmetic.
  if (n == 0 || n > max_rows_ || total != n ||
      (segments.size() > 1 && n > kSplitRows) ||
      logit_rows.size() > max_logit_rows_) {
    throw std::invalid_argument("gemma4 forward exceeds its capacity");
  }
  if (fences_ != nullptr) {
    writers_.clear();
    for (const Segment& s : segments) {
      if (s.cache->hooks != nullptr) {
        writers_.push_back(s.cache->hooks);
      }
    }
    fences_->BeforeMutation(writers_);
  }
  for (const Segment& s : segments) {
    if (s.cache->hooks != nullptr) {
      s.cache->hooks->BeforeWrite(s.first_position, s.first_position + s.rows);
    }
  }
  const std::uint32_t d = c.hidden_size;
  const float eps = c.rms_eps;
  const auto& layers = model_.layers();
  HIP_CHECK(hipMemcpyAsync(tokens_, tokens.data(), n * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream_));
  hip::LaunchBatchedEmbeddingLookup(
      model_.token_embd().data, model_.token_embd().type,
      reinterpret_cast<const std::uint32_t*>(tokens_), x_, n, d, stream_);
  ScaleRmsNorm(x_, std::sqrt(static_cast<float>(d)), layers[0].attn_norm.f32(),
               h_, n, d, eps, stream_);
  // Image rows take their embeddings unscaled; each image's rows see all of
  // its keys in sliding layers.
  const std::uint32_t* key_ends = nullptr;
  if (!images.empty()) {
    if (segments.size() != 1) {
      throw std::invalid_argument(
          "gemma4 images need a single-session forward");
    }
    key_ends_host_.assign(n, 0);
    std::uint32_t previous_end = 0;
    for (const ImageRows& image : images) {
      if (image.count == 0 || image.row < previous_end ||
          image.count > n - image.row || image.embedding == nullptr) {
        throw std::invalid_argument("gemma4 image rows are out of order");
      }
      previous_end = image.row + image.count;
      HIP_CHECK(hipMemcpyAsync(x_ + std::size_t{image.row} * d, image.embedding,
                               std::size_t{image.count} * d * sizeof(float),
                               hipMemcpyDeviceToDevice, stream_));
      RmsNorm(x_ + std::size_t{image.row} * d, layers[0].attn_norm.f32(),
              h_ + std::size_t{image.row} * d, image.count, d, eps, stream_);
      std::fill_n(key_ends_host_.begin() + image.row, image.count,
                  segments[0].first_position + previous_end);
    }
    HIP_CHECK(hipMemcpyAsync(key_ends_, key_ends_host_.data(),
                             n * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                             stream_));
    key_ends = key_ends_;
  }

  // Prefill (rows past the small-batch width) quantizes activations for the
  // W8A8 GEMMs; the norms and GeGLU write that encoding directly. Binary16
  // prefill stages each input before its projections instead.
  const bool prefill = n > kSplitRows && !half_prefill_;
  // Binary16 prefill: after the first layer's input, the norms, attention,
  // GeGLU and mixture write each projection input as binary16 themselves.
  const bool half = n > kSplitRows && half_prefill_;
  static_assert(kMaxGroupSlots == kSplitRows,
                "binary16 prefill experts read the expert input the norm "
                "writes past the grouped width");
  // Whether the previous layer's mixture wrote this layer's binary16 input.
  bool half_ready = false;
  // Whether h_ (the current layer's input) already holds its WMMA pack.
  bool h_packed = false;
  // A binary16 input whose every consumer is a Q8_0 or binary16 projection
  // (the binary16 GEMMs) needs no FP32 row.
  const auto half_only = [&](std::initializer_list<const DeviceTensor*> ws) {
    return half && std::all_of(ws.begin(), ws.end(), [](const DeviceTensor* w) {
             return w->empty() || HalfPrefillFormat(w->type);
           });
  };
  const auto qkv_half_only = [&](const DeviceLayer& L) {
    return !L.attn_qkv.empty() ? half_only({&L.attn_qkv, &L.attn_v})
                               : half_only({&L.attn_q, &L.attn_k, &L.attn_v});
  };
  // A W8A8 prefill input that a binary16 projection reads keeps its FP32
  // rows (the norms write Q8_1 rows instead of them); its Q8_0 consumers
  // quantize those rows.
  const auto fp32_input = [&](std::initializer_list<const DeviceTensor*> ws) {
    return prefill &&
           std::any_of(ws.begin(), ws.end(), [](const DeviceTensor* w) {
             return w->type == core::GgmlType::kF16;
           });
  };
  const auto qkv_fp32 = [&](const DeviceLayer& L) {
    return fp32_input({&L.attn_qkv, &L.attn_q, &L.attn_k, &L.attn_v});
  };
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const DeviceLayer& L = layers[l];
    const bool sliding = c.IsSliding(l);
    const std::uint32_t dim = c.HeadDim(l);
    // Layer l > 0 of a prefill reads the rows the previous layer's post-FFN
    // norm wrote (Q8_1 or binary16).
    const void* hq = l == 0                    ? Quantize(h_, n, d)
                     : half_ready              ? x_half_
                     : prefill && !qkv_fp32(L) ? q8_
                                               : Quantize(h_, n, d);
    // A fused projection leaves [q | k | v] or [q | k] rows in k_; QkvPost
    // reads them strided and writes packed queries to q_. At WMMA decode
    // widths [q | k] stays split (64 and 32 blocks fit one wave of blocks
    // each; 96 spills into a second), and the split pair measures faster: a
    // fused 12288-row launch of 96 eight-wave blocks walks at 180 GB/s against
    // 190 GB/s for the pair, MTP tg128 d0 31.41 -> 31.21 tok/s.
    const bool wmma_qkv =
        n <= kSplitRows && (model_.has_draft() || n >= 6) &&
        WmmaGemvSupports(L.attn_qkv.type, L.attn_qkv.rows, L.attn_qkv.cols);
    const bool separate_v =
        !L.attn_v.empty() && L.attn_qkv.rows == c.QDim(l) + c.KvDim(l);
    const std::uint32_t qkv_stride =
        L.attn_qkv.empty() || (separate_v && wmma_qkv) ? 0 : L.attn_qkv.rows;
    float* q_source = q_;
    float* k_source = k_;
    float* v_source = k_;
    // h_packed: the previous layer's norm left h_'s pack (width d); the
    // first WMMA projection over h_ packs it otherwise, for the rest.
    if (qkv_stride != 0) {
      Project(L.attn_qkv, h_, hq, n, k_, h_packed);
      q_source = k_;
      k_source = k_ + c.QDim(l);
      if (separate_v) {
        v_source = k_ + std::size_t{n} * qkv_stride;
        Project(L.attn_v, h_, hq, n, v_source,
                h_packed || WmmaProjection(L.attn_qkv, n));
      } else {
        v_source = L.attn_v.empty() ? k_source : k_source + c.KvDim(l);
      }
    } else {
      Project(L.attn_q, h_, hq, n, q_, h_packed);
      const bool q_packed = h_packed || WmmaProjection(L.attn_q, n);
      Project(L.attn_k, h_, hq, n, k_, q_packed);
      if (!L.attn_v.empty()) {
        Project(L.attn_v, h_, hq, n, v_,
                q_packed || WmmaProjection(L.attn_k, n));
        v_source = v_;
      }
    }
    const std::size_t q_row = qkv_stride != 0 ? qkv_stride : c.QDim(l);
    const std::size_t kv_row = qkv_stride != 0 ? qkv_stride : c.KvDim(l);
    const std::size_t v_row =
        qkv_stride != 0 && !separate_v ? qkv_stride : c.KvDim(l);
    // At decode widths the attention merge packs its FP32 rows for a WMMA
    // output projection.
    const WmmaPackOut attn_pack = prefill || half_only({&L.attn_output})
                                      ? WmmaPackOut{}
                                      : PackFor(L.attn_output, n);
    std::uint32_t row0 = 0;
    for (const Segment& seg : segments) {
      KvCache& cache = *seg.cache;
      QkvPostArgs post{};
      post.q = q_source + row0 * q_row;
      post.k = k_source + row0 * kv_row;
      post.v = v_source + row0 * v_row;
      post.row_stride = qkv_stride;
      post.v_stride = separate_v ? 0 : qkv_stride;
      post.q_out =
          qkv_stride != 0 ? q_ + std::size_t{row0} * c.QDim(l) : nullptr;
      post.q_norm = L.attn_q_norm.f32();
      post.k_norm = L.attn_k_norm.f32();
      post.theta_scale =
          std::pow(c.RopeTheta(l), -2.0F / static_cast<float>(dim));
      post.freq_factors = sliding ? nullptr : model_.rope_factors();
      post.k_cache = cache.k[l];
      post.v_cache = cache.v[l];
      post.rows = seg.rows;
      post.heads = c.num_heads;
      post.kv_heads = c.kv_heads[l];
      post.head_dim = dim;
      post.first_position = seg.first_position;
      post.ring = sliding ? cache.ring : 0;
      post.eps = eps;
      post.rotated_pairs = DerivedKeys(l) ? model_.global_rope_pairs() : 0;
      post.siblings = seg.siblings;
      post.spare_key = seg.first_position + seg.rows - seg.siblings;
      QkvPost(post, stream_);

      AttentionArgs att{};
      att.q = q_ + std::size_t{row0} * c.QDim(l);
      att.k_cache = cache.k[l];
      att.v_cache = cache.v[l];
      att.out = half_only({&L.attn_output})
                    ? nullptr
                    : attn_ + std::size_t{row0} * c.QDim(l);
      att.partials = partials_;
      att.rows = seg.rows;
      att.heads = c.num_heads;
      att.kv_heads = c.kv_heads[l];
      att.head_dim = dim;
      att.first_position = seg.first_position;
      att.shared_position = false;
      att.key_limit = std::numeric_limits<std::uint32_t>::max();
      att.window = sliding ? c.sliding_window : 0;
      att.ring = sliding ? cache.ring : 0;
      att.key_ends = sliding ? key_ends : nullptr;
      att.out_half =
          half ? static_cast<__half*>(x_half_) + std::size_t{row0} * c.QDim(l)
               : nullptr;
      att.siblings = post.siblings;
      att.spare_key = post.spare_key;
      att.pack = attn_pack;
      att.pack_row0 = row0;
      DeriveKeys(att, l);
      Attention(att, stream_);
      row0 += seg.rows;
    }

    Project(L.attn_output, attn_,
            half ? x_half_ : Quantize(attn_, n, c.QDim(l)), n, o_,
            attn_pack.hi != nullptr);
    // Expert layers also normalize the attention residual for the experts.
    const bool moe = !L.gate_up_exps.empty();
    const bool dense_half_only = !L.ffn_gate_up.empty()
                                     ? half_only({&L.ffn_gate_up})
                                     : half_only({&L.ffn_gate, &L.ffn_up});
    const bool experts_half_only = half && moe && HalfExperts(L, c);
    const bool dense_fp32 =
        fp32_input({&L.ffn_gate_up, &L.ffn_gate, &L.ffn_up});
    // At decode widths the norm packs h_ for a WMMA [gate | up] (or gate).
    const WmmaPackOut ffn_pack =
        prefill || dense_half_only
            ? WmmaPackOut{}
            : PackFor(!L.ffn_gate_up.empty() ? L.ffn_gate_up : L.ffn_gate, n);
    PostAttentionNorm(o_, L.post_attn_norm.f32(), x_, L.ffn_norm.f32(),
                      dense_half_only ? nullptr : h_, n, d, eps, stream_,
                      prefill && !dense_fp32 ? q8_ : nullptr,
                      moe ? L.pre_ffn_norm_2.f32() : nullptr,
                      moe && !experts_half_only ? moe_h_ : nullptr,
                      half ? x_half_ : nullptr,
                      half && moe ? moe_x_half_ : nullptr, ffn_pack);
    const bool ffn_packed = ffn_pack.hi != nullptr;
    const void* fq = prefill && !dense_fp32 ? q8_
                     : half                 ? x_half_
                                            : Quantize(h_, n, d);
    // At grouped widths the experts run beside the dense MLP on their own
    // stream: routing is latency bound, the projections bandwidth bound.
    const bool concurrent = moe && n <= kMaxGroupSlots;
    if (concurrent) {
      HIP_CHECK(hipEventRecord(expert_fork_, stream_));
      HIP_CHECK(hipStreamWaitEvent(expert_stream_, expert_fork_, 0));
      Experts(L, n, expert_stream_);
      HIP_CHECK(hipEventRecord(expert_join_, expert_stream_));
    }
    const void* gq = nullptr;
    const float* down_in = gate_;
    WmmaPackOut down_pack{};
    const auto gate_up_half = DenseHalfFormat(L.ffn_gate_up.type);
    const bool gate_up_packed = PackedPrefill(L.ffn_gate_up, n);
    const bool gate_up_q8 = L.ffn_gate_up.type == core::GgmlType::kQ8_0;
    const bool gate_up_f16 = L.ffn_gate_up.type == core::GgmlType::kF16;
    if (!L.ffn_gate_up.empty() && half &&
        (gate_up_half || gate_up_packed || gate_up_q8 || gate_up_f16) &&
        half_only({&L.ffn_down}) &&
        ((gate_up_packed &&
          LaunchPackedPrefillGemm(
              L.ffn_gate_up.prefill_format, L.ffn_gate_up.prefill_layout,
              L.ffn_gate_up.prefill, fq, nullptr, gate_, n, L.ffn_gate_up.rows,
              L.ffn_gate_up.cols, stream_, true)) ||
         (gate_up_half &&
          LaunchDenseHalfGemm(*gate_up_half, L.ffn_gate_up.data, fq, nullptr,
                              gate_, n, L.ffn_gate_up.rows, L.ffn_gate_up.cols,
                              stream_, true, L.ffn_gate_up.grouped)) ||
         // Q8_0 and binary16: GeGLU of the FP32 [gate | up] sums, as the
         // separate pass.
         ((gate_up_q8 || gate_up_f16) && !gate_up_packed &&
          qwen38_flash_next::rocm::DenseF16GeGluGemm(
              L.ffn_gate_up.data, static_cast<const __half*>(fq),
              static_cast<__half*>(static_cast<void*>(gate_)), n,
              L.ffn_gate_up.rows, L.ffn_gate_up.cols, stream_,
              L.ffn_gate_up.grouped, gate_up_f16)))) {
      // The GEMM writes GeGLU of each binary16 [gate | up] pair into the
      // (free) gate rows as the down projection's binary16 input.
      gq = gate_;
    } else if (!L.ffn_gate_up.empty() && !prefill) {
      // One projection writes [gate | up] rows; GeGLU packs its rows for a
      // WMMA down projection.
      Project(L.ffn_gate_up, h_, fq, n, gate_, ffn_packed);
      const bool act_fp32 = !half_only({&L.ffn_down});
      down_pack = act_fp32 ? PackFor(L.ffn_down, n) : WmmaPackOut{};
      GeGluPacked(gate_, act_fp32 ? act_ : nullptr, n, c.ffn_size, stream_,
                  half ? x_half_ : nullptr, down_pack);
      down_in = act_;
      gq = half ? x_half_ : Quantize(act_, n, c.ffn_size);
    } else {
      Project(L.ffn_gate, h_, fq, n, gate_, ffn_packed);
      Project(L.ffn_up, h_, fq, n, up_,
              ffn_packed || WmmaProjection(L.ffn_gate, n));
      // A binary16 down projection narrows the FP32 GeGLU rows itself.
      if (prefill && L.ffn_down.type != core::GgmlType::kF16) {
        GeGluQuantize(gate_, up_, q8_, n, c.ffn_size, stream_);
        gq = q8_;
      } else {
        GeGlu(gate_, up_, gate_, std::size_t{n} * c.ffn_size, stream_);
        gq = Quantize(gate_, n, c.ffn_size);
      }
    }
    Project(L.ffn_down, down_in, gq, n, o_, down_pack.hi != nullptr);
    const bool last = l + 1 == c.num_layers;
    // At decode widths the layer's last norm packs the next layer's input for
    // its WMMA Q/K/V projections (the vocabulary head after the last layer
    // packs its gathered rows itself).
    WmmaPackOut next_pack{};
    if (!last && !prefill) {
      const DeviceLayer& N = layers[l + 1];
      if (WmmaProjection(N.attn_q, n) ||
          (!N.attn_qkv.empty() && WmmaProjection(N.attn_qkv, n))) {
        next_pack = WmmaPackOf(wmma_, d);
      }
    }
    bool next_h_packed = false;
    const float* next =
        !last ? layers[l + 1].attn_norm.f32() : model_.output_norm().f32();
    // The output-normed rows of the last layer feed the vocabulary head and
    // the drafter in FP32.
    if (moe) {
      if (concurrent) {
        HIP_CHECK(hipStreamWaitEvent(stream_, expert_join_, 0));
      } else {
        Experts(L, n, stream_);
      }
      MoeFinishArgs finish{};
      finish.dense = o_;
      finish.experts = moe_out_;
      finish.weights = moe_weights_;
      finish.norm1 = L.post_ffn_norm_1.f32();
      finish.norm2 = L.post_ffn_norm_2.f32();
      finish.post_norm = L.post_ffn_norm.f32();
      finish.scale = L.output_scale;
      finish.x = x_;
      finish.next_norm = next;
      finish.h = half && !last && qkv_half_only(layers[l + 1]) ? nullptr : h_;
      finish.rows = n;
      finish.hidden = d;
      finish.used = c.experts_used;
      finish.eps = eps;
      finish.h_half = half && !last ? x_half_ : nullptr;
      finish.pack = next_pack;
      MoeFinish(finish, stream_);
      half_ready = finish.h_half != nullptr;
      next_h_packed = next_pack.hi != nullptr;
    } else {
      // Binary16 prefill: the norm also writes the next layer's binary16
      // input (and only that when every consumer reads binary16).
      const bool next_half = half && !last;
      PostFeedForwardNorm(
          o_, L.post_ffn_norm.f32(), L.output_scale, x_, next,
          next_half && qkv_half_only(layers[l + 1]) ? nullptr : h_, n, d, eps,
          stream_, prefill && !last && !qkv_fp32(layers[l + 1]) ? q8_ : nullptr,
          next_half ? x_half_ : nullptr, next_pack);
      half_ready = next_half;
      next_h_packed = next_pack.hi != nullptr;
    }
    h_packed = next_h_packed;
    if (tap_sink_) {
      tap_sink_(l, x_, n, stream_);
    }
  }

  const auto m = static_cast<std::uint32_t>(logit_rows.size());
  if (m == 0) {
    return;
  }
  bool contiguous = true;
  for (std::uint32_t i = 1; i < m; ++i) {
    contiguous = contiguous && logit_rows[i] == logit_rows[0] + i;
  }
  const float* h_logits = h_ + std::size_t{logit_rows[0]} * d;
  if (!contiguous) {
    HIP_CHECK(hipMemcpyAsync(logit_index_, logit_rows.data(),
                             m * sizeof(std::uint32_t), hipMemcpyHostToDevice,
                             stream_));
    GatherRows(h_, logit_index_, hsel_, m, d, stream_);
    h_logits = hsel_;
  }
  Project(model_.output(), h_logits, Quantize(h_logits, m, d), m, logits_);
  if (c.final_logit_softcap > 0.0F) {
    Softcap(logits_, std::size_t{m} * model_.vocab_size(),
            c.final_logit_softcap, stream_);
  }
}

namespace {

std::optional<ExpertFormat> ExpertFormatOf(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      return ExpertFormat::kQ4_K;
    case core::GgmlType::kQ5_K:
      return ExpertFormat::kQ5_K;
    case core::GgmlType::kQ6_K:
      return ExpertFormat::kQ6_K;
    case core::GgmlType::kQ8_0:
      return ExpertFormat::kQ8_0;
    case core::GgmlType::kQ5_1:
      return ExpertFormat::kQ5_1;
    case core::GgmlType::kF16:
      return ExpertFormat::kF16;
    case core::GgmlType::kQ4_0:
      return ExpertFormat::kQ4_0;
    case core::GgmlType::kQ5_0:
      return ExpertFormat::kQ5_0;
    default:
      return std::nullopt;
  }
}

/// Flash-Next's routed binary16 GEMM formats.
std::optional<qwen38_flash_next::rocm::WeightType> RoutedHalfType(
    core::GgmlType type) {
  using qwen38_flash_next::rocm::WeightType;
  switch (type) {
    case core::GgmlType::kQ4_K:
      return WeightType::kQ4_K;
    case core::GgmlType::kQ5_K:
      return WeightType::kQ5_K;
    case core::GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    case core::GgmlType::kQ5_1:
      return WeightType::kQ5_1;
    default:
      return std::nullopt;
  }
}

/// Gate/up formats the Gemma routed binary16 GEMM takes (with its GeGLU
/// epilogue).
bool OwnGateUp(std::optional<ExpertFormat> format) {
  return format &&
         (*format == ExpertFormat::kQ4_K || *format == ExpertFormat::kQ5_K ||
          *format == ExpertFormat::kQ6_K || *format == ExpertFormat::kQ8_0 ||
          *format == ExpertFormat::kF16 || *format == ExpertFormat::kQ4_0);
}

/// Down formats the Gemma routed binary16 GEMM takes.
bool OwnDown(std::optional<ExpertFormat> format) {
  return format &&
         (*format == ExpertFormat::kQ5_1 || *format == ExpertFormat::kQ8_0 ||
          *format == ExpertFormat::kF16 || *format == ExpertFormat::kQ4_0 ||
          *format == ExpertFormat::kQ5_0);
}

bool HalfExperts(const DeviceLayer& l, const Config& c) {
  return (RoutedHalfType(l.gate_up_exps.type) ||
          OwnGateUp(ExpertFormatOf(l.gate_up_exps.type))) &&
         (RoutedHalfType(l.down_exps.type) ||
          OwnDown(ExpertFormatOf(l.down_exps.type))) &&
         c.hidden_size % 256 == 0 && c.expert_ffn_size % 64 == 0;
}

}  // namespace

void Executor::Experts(const DeviceLayer& l, std::uint32_t n,
                       hipStream_t stream) {
  const Config& c = model_.config();
  const std::uint32_t d = c.hidden_size;
  const std::uint32_t used = c.experts_used;
  const std::uint32_t width = c.expert_ffn_size;
  const bool grouped = n <= kMaxGroupSlots;
  if (!grouped) {
    HIP_CHECK(hipMemsetAsync(moe_counts_, 0,
                             c.num_experts * sizeof(std::uint32_t), stream));
  }
  MoeRouteArgs route{};
  route.x = x_;
  route.router = l.router.f32();
  route.expert_scale = l.down_exps_scale.f32();
  route.logits = moe_logits_;
  route.ids = moe_ids_;
  route.weights = moe_weights_;
  route.counts = grouped ? nullptr : moe_counts_;
  route.groups = grouped ? moe_groups_ : nullptr;
  route.sync = grouped ? moe_sync_ : nullptr;
  route.rows = n;
  route.hidden = d;
  route.experts = c.num_experts;
  route.used = used;
  route.eps = c.rms_eps;
  if (!grouped) {
    // Prefill logits as one GEMM: logits[r][e] = router[e] . x[r].
    const float alpha = 1.0F;
    const float beta = 0.0F;
    if (hipblasSgemm(
            blas_, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(c.num_experts),
            static_cast<int>(n), static_cast<int>(d), &alpha, l.router.f32(),
            static_cast<int>(d), x_, static_cast<int>(d), &beta, moe_logits_,
            static_cast<int>(c.num_experts)) != HIPBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("gemma4 router GEMM failed");
    }
    route.raw_logits = true;
  }
  MoeRoute(route, stream);
  route.raw_logits = false;
  if (grouped || !PrefillExperts(l, n, stream)) {
    // Grouped FP32 projections; a prefill whose formats lack the binary16
    // route runs them over 16-row pieces.
    const auto gate_up = ExpertFormatOf(l.gate_up_exps.type);
    const auto down = ExpertFormatOf(l.down_exps.type);
    for (std::uint32_t r0 = 0; r0 < n; r0 += kMaxGroupSlots) {
      const std::uint32_t rows = std::min(kMaxGroupSlots, n - r0);
      if (!grouped) {
        route.x = x_ + std::size_t{r0} * d;
        route.logits = moe_logits_ + std::size_t{r0} * c.num_experts;
        route.ids = moe_ids_ + std::size_t{r0} * used;
        route.weights = moe_weights_ + std::size_t{r0} * used;
        route.counts = nullptr;
        route.groups = moe_groups_;
        route.rows = rows;
        MoeRoute(route, stream);
      }
      const std::uint32_t max_groups = std::min(c.num_experts, rows * used);
      auto* act = static_cast<float*>(moe_act_);
      // The gate/up projection applies GeGLU in its epilogue.
      if (!gate_up || !down ||
          !LaunchRoutedGemv(*gate_up, l.gate_up_exps.data, moe_groups_,
                            max_groups, moe_h_ + std::size_t{r0} * d, used, act,
                            2 * width, d, stream, true, rows == 1)) {
        throw std::runtime_error("gemma4 expert gate/up format unsupported");
      }
      // One row: every expert group holds a single slot.
      if (!LaunchRoutedGemv(*down, l.down_exps.data, moe_groups_, max_groups,
                            act, 1, moe_out_ + std::size_t{r0} * used * d, d,
                            width, stream, false, rows == 1)) {
        throw std::runtime_error("gemma4 expert down format unsupported");
      }
    }
  }
}

bool Executor::PrefillExperts(const DeviceLayer& l, std::uint32_t n,
                              hipStream_t stream) {
  namespace fn = qwen38_flash_next::rocm;
  const Config& c = model_.config();
  const std::uint32_t d = c.hidden_size;
  const std::uint32_t used = c.experts_used;
  const std::uint32_t width = c.expert_ffn_size;
  if (!HalfExperts(l, c)) {
    return false;
  }
  const auto gate_up = RoutedHalfType(l.gate_up_exps.type);
  const auto down = RoutedHalfType(l.down_exps.type);
  const auto own_format = ExpertFormatOf(l.gate_up_exps.type);
  const bool own_gate_up = OwnGateUp(own_format);
  // Tile maps built on the device, the down projection's after the gate/up
  // one; each GEMM launches its map's capacity and skips the dead entries.
  const std::uint32_t slots = n * used;
  const std::uint32_t gate_up_rows =
      own_gate_up ? kRoutedKQuantTileRows : kRoutedTileRows;
  const std::uint32_t tiles =
      RoutedTileCapacity(slots, c.num_experts, gate_up_rows);
  // The down projection takes the Gemma routed GEMM (Q4_0, Q5_0, Q5_1, Q8_0,
  // binary16).
  const auto down_format = ExpertFormatOf(l.down_exps.type);
  const bool own_down = OwnDown(down_format);
  const std::uint32_t down_rows =
      own_down ? kRoutedKQuantTileRows : kRoutedDownRows;
  const std::uint32_t down_tiles =
      RoutedTileCapacity(slots, c.num_experts, down_rows);
  std::int32_t* down_map = moe_tiles_ + tiles;
  BuildRoutedTiles(moe_counts_, c.num_experts, gate_up_rows, tiles, moe_tiles_,
                   down_rows, down_tiles, down_map, stream);
  fn::RoutedCompact(moe_ids_, moe_counts_, moe_bounds_, moe_cursors_,
                    moe_rows_token_, moe_rows_slot_, n, used, c.num_experts,
                    stream);
  // PostAttentionNorm wrote the binary16 expert input.
  const void* x_half = moe_x_half_;
  auto* gu = static_cast<__half*>(moe_gu_);
  auto* act = static_cast<__half*>(moe_act_);
  // The Gemma routed GEMM applies GeGLU in its epilogue; Flash-Next's writes
  // [gate | up] for the separate pass.
  if (own_gate_up) {
    const bool compact =
        l.gate_up_exps.prefill_layout == PrefillLayout::kCompact;
    if (!LaunchRoutedHalfGemm(
            *own_format, compact ? l.gate_up_exps.prefill : l.gate_up_exps.data,
            x_half, moe_tiles_, tiles, gate_up_rows, moe_bounds_,
            moe_rows_token_, moe_rows_slot_, nullptr, act, 2 * width, d, stream,
            true, l.gate_up_exps.prefill_layout)) {
      return false;
    }
  } else {
    if (!fn::RoutedF16Gemm(
            l.gate_up_exps.data, *gate_up, static_cast<const __half*>(x_half),
            moe_tiles_, tiles, kRoutedTileRows, moe_bounds_, moe_rows_token_,
            moe_rows_slot_, nullptr, nullptr, gu, 2 * width, d, stream)) {
      return false;
    }
    GeGluPackedHalf(gu, act, n * used, width, stream);
  }
  if (own_down) {
    return LaunchRoutedHalfGemm(*down_format, l.down_exps.data, act, down_map,
                                down_tiles, down_rows, moe_bounds_,
                                moe_rows_slot_, moe_rows_slot_, moe_out_,
                                nullptr, d, width, stream);
  }
  return fn::RoutedF16Gemm(l.down_exps.data, *down, act, down_map, down_tiles,
                           kRoutedDownRows, moe_bounds_, moe_rows_slot_,
                           moe_rows_slot_, nullptr, moe_out_, nullptr, d, width,
                           stream);
}

void Executor::BeforeMutation(KvCache& cache) {
  if (fences_ != nullptr && cache.hooks != nullptr) {
    ContinuationHooks* const writer = cache.hooks;
    fences_->BeforeMutation(std::span(&writer, 1));
  }
}

void Executor::CommitHidden(KvCache& cache, std::uint32_t row) {
  BeforeMutation(cache);
  const std::size_t d = model_.config().hidden_size;
  HIP_CHECK(hipMemcpyAsync(cache.hidden, h_ + row * d, d * sizeof(float),
                           hipMemcpyDeviceToDevice, stream_));
}

void Executor::StashHidden(KvCache& cache, std::uint32_t row,
                           std::uint32_t slot) {
  if (slot >= kSplitRows) {
    throw std::invalid_argument("gemma4 hidden stash slot");
  }
  const std::size_t d = model_.config().hidden_size;
  HIP_CHECK(hipMemcpyAsync(cache.ahead_hidden + slot * d, h_ + row * d,
                           d * sizeof(float), hipMemcpyDeviceToDevice,
                           stream_));
}

void Executor::AdoptStashedHidden(KvCache& cache, std::uint32_t slot) {
  if (slot >= kSplitRows) {
    throw std::invalid_argument("gemma4 hidden stash slot");
  }
  BeforeMutation(cache);
  const std::size_t d = model_.config().hidden_size;
  HIP_CHECK(hipMemcpyAsync(cache.hidden, cache.ahead_hidden + slot * d,
                           d * sizeof(float), hipMemcpyDeviceToDevice,
                           stream_));
}

void Executor::MoveKey(KvCache& cache, std::uint32_t from, std::uint32_t to) {
  const Config& c = model_.config();
  MoveKeyArgs args{};
  if (c.num_layers > MoveKeyArgs::kMaxLayers) {
    throw std::invalid_argument("gemma4 MoveKey layer count");
  }
  args.layers = c.num_layers;
  args.from = from;
  args.to = to;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    args.k[l] = cache.k[l];
    args.v[l] = cache.v[l];
    args.k_width[l] = key_widths_[l];
    args.v_width[l] = c.KvDim(l);
    args.ring[l] = c.IsSliding(l) ? cache.ring : 0;
  }
  BeforeMutation(cache);
  if (cache.hooks != nullptr) {
    cache.hooks->BeforeWrite(to, to + 1);
  }
  cache.written_from = std::min(cache.written_from, to);
  cache.written_to = std::max(cache.written_to, to + 1);
  rocm::MoveKey(args, stream_);
}

void Executor::CopyRuns(std::span<const CopyRun> runs) {
  std::uint64_t longest = 0;
  for (const CopyRun& run : runs) {
    if (run.bytes % 16 != 0 || (reinterpret_cast<std::uintptr_t>(run.from) |
                                reinterpret_cast<std::uintptr_t>(run.to)) %
                                       16 !=
                                   0) {
      throw std::invalid_argument("gemma4 copy run alignment");
    }
    longest = std::max(longest, run.bytes);
  }
  if (runs.size() > copy_runs_capacity_) {
    if (copy_runs_ != nullptr) {
      HIP_CHECK(hipStreamSynchronize(stream_));
      HIP_CHECK(hipFree(copy_runs_));
      copy_runs_ = nullptr;
      copy_runs_capacity_ = 0;
    }
    const std::size_t capacity = std::bit_ceil(runs.size());
    HIP_CHECK(hip::AllocateDevice(&copy_runs_, capacity * sizeof(CopyRun)));
    copy_runs_capacity_ = capacity;
  }
  if (runs.empty()) {
    return;
  }
  HIP_CHECK(hipMemcpyAsync(copy_runs_, runs.data(), runs.size_bytes(),
                           hipMemcpyHostToDevice, stream_));
  rocm::CopyRuns(copy_runs_, static_cast<std::uint32_t>(runs.size()), longest,
                 stream_);
}

void Executor::DraftChain(KvCache& cache, std::int32_t token,
                          std::uint32_t position, std::uint32_t steps,
                          std::vector<std::int32_t>* drafts,
                          const DraftProposer& propose, DraftHead head) {
  DraftJob job{&cache, token, position, steps, propose, drafts, head};
  DraftChains(std::span<DraftJob>(&job, 1));
}

void Executor::LoadDraftTable(KvCache& cache, std::uint32_t first,
                              std::span<const std::int32_t> tokens) {
  if (tokens.empty()) {
    return;
  }
  const auto rows = model_.draft_head_rows();
  if (cache.draft_table == nullptr || first + tokens.size() > kDraftTableRows) {
    throw std::invalid_argument("gemma4 draft table exceeds its capacity");
  }
  const std::size_t row_bytes = model_.draft_row_bytes();
  const auto* head =
      static_cast<const std::uint8_t*>(model_.draft().token_embd.data);
  auto* table = static_cast<std::uint8_t*>(cache.draft_table);
  std::vector<CopyRun> runs(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    runs[i] = {head + rows[static_cast<std::size_t>(tokens[i])] * row_bytes,
               table + (first + i) * row_bytes, row_bytes};
  }
  CopyRuns(runs);
}

void Executor::DraftChains(std::span<DraftJob> jobs) {
  const auto n = static_cast<std::uint32_t>(jobs.size());
  std::uint32_t max_steps = 0;
  for (const DraftJob& job : jobs) {
    if (job.steps > kMaxDraftTokens || !job.propose || job.drafts == nullptr) {
      throw std::invalid_argument("gemma4 draft chain exceeds its capacity");
    }
    max_steps = std::max(max_steps, job.steps);
  }
  if (!model_.has_draft() || n == 0 || n > kMaxDraftSessions) {
    throw std::invalid_argument("gemma4 draft chain exceeds its capacity");
  }
  const Config& tc = model_.config();
  const DeviceDraft& dm = model_.draft();
  const Config& c = dm.config;
  const std::size_t target_hidden = tc.hidden_size;
  const std::size_t hidden = c.hidden_size;
  const float eps = c.rms_eps;
  const float embed_scale = std::sqrt(static_cast<float>(target_hidden));
  const std::uint32_t vocab = model_.vocab_size();
  using qwen38_flash_next::kMtpCandidates;
  using qwen38_flash_next::MtpCandidateLogits;
  const std::size_t candidate_ids =
      qwen38_flash_next::rocm::MtpCandidateWorkspaceSize(vocab);
  const std::span<const std::int32_t> head_tokens = model_.draft_head_tokens();
  const bool small = n == 1 && jobs[0].head.small && !head_tokens.empty() &&
                     GemvFormatOf(dm.token_embd.type).has_value() &&
                     dm.token_embd.rows > kDraftHeadRows &&
                     jobs[0].head.table.size() <= kDraftTableRows;
  const std::span<const std::int32_t> table =
      small ? jobs[0].head.table : std::span<const std::int32_t>{};
  // Row j drafts for job j every step; rows whose chain ended keep being
  // projected (the batched projections cost the same) but are not attended,
  // ranked or proposed.
  std::vector<std::uint8_t> active(n);
  std::vector<std::int32_t> tokens(n);
  for (std::uint32_t j = 0; j < n; ++j) {
    jobs[j].drafts->clear();
    jobs[j].drafts->reserve(jobs[j].steps);
    active[j] = jobs[j].steps > 0 ? 1 : 0;
    tokens[j] = jobs[j].token;
  }
  HIP_CHECK(hipMemcpyAsync(draft_tokens_, tokens.data(),
                           n * sizeof(std::int32_t), hipMemcpyHostToDevice,
                           stream_));
  for (std::uint32_t step = 0; step < max_steps; ++step) {
    if (std::find(active.begin(), active.end(), 1) == active.end()) {
      break;
    }
    // [scaled target embedding of the token ; target-width hidden state]
    hip::LaunchBatchedEmbeddingLookup(model_.token_embd().data,
                                      model_.token_embd().type, draft_tokens_,
                                      draft_embed_, n, target_hidden, stream_);
    DraftConcatArgs concat{};
    concat.embed = draft_embed_;
    concat.out = draft_concat_;
    concat.scale = embed_scale;
    concat.rows = n;
    concat.dim = static_cast<std::uint32_t>(target_hidden);
    for (std::uint32_t j = 0; j < n; ++j) {
      concat.hidden[j] =
          step == 0 ? jobs[j].cache->hidden : draft_next_ + j * target_hidden;
    }
    DraftConcat(concat, stream_);
    ProjectDraft(dm.pre_projection, draft_concat_, n, draft_x_);
    RmsNorm(draft_x_, dm.layers[0].attn_norm.f32(), draft_h_, n, hidden, eps,
            stream_);
    for (std::uint32_t l = 0; l < c.num_layers; ++l) {
      const DeviceLayer& L = dm.layers[l];
      const bool sliding = c.IsSliding(l);
      const std::uint32_t dim = c.HeadDim(l);
      const std::uint32_t q_dim = c.QDim(l);
      const std::uint32_t source = c.SharedKvSource(l, tc);
      ProjectDraft(L.attn_q, draft_h_, n, draft_q_);
      for (std::uint32_t j = 0; j < n; ++j) {
        if (active[j] == 0) {
          continue;
        }
        KvCache& cache = *jobs[j].cache;
        const std::uint32_t position = jobs[j].position;
        float* q = draft_q_ + std::size_t{j} * q_dim;
        QueryPost(
            q, L.attn_q_norm.f32(),
            std::pow(c.RopeTheta(l), -2.0F / static_cast<float>(dim)),
            sliding ? nullptr : dm.rope_factors, 1, c.num_heads, dim, position,
            true, eps, model_.layers()[source].attn_k_norm.f32(),
            DerivedKeys(source) ? model_.global_rope_pairs() : 0, stream_);
        AttentionArgs att{};
        att.q = q;
        att.k_cache = cache.k[source];
        att.v_cache = cache.v[source];
        att.out = draft_attn_ + std::size_t{j} * q_dim;
        att.partials = partials_;
        att.rows = 1;
        att.heads = c.num_heads;
        att.kv_heads = c.kv_heads[l];
        att.head_dim = dim;
        att.first_position = position;
        att.shared_position = true;
        att.key_limit = position;  // committed keys only
        att.window = sliding ? c.sliding_window : 0;
        att.ring = sliding ? cache.ring : 0;
        DeriveKeys(att, source);
        Attention(att, stream_);
      }
      ProjectDraft(L.attn_output, draft_attn_, n, draft_o_);
      PostAttentionNorm(draft_o_, L.post_attn_norm.f32(), draft_x_,
                        L.ffn_norm.f32(), draft_h_, n, hidden, eps, stream_);
      if (!L.ffn_gate_up.empty()) {
        ProjectDraft(L.ffn_gate_up, draft_h_, n, draft_gate_);
        GeGluPacked(draft_gate_, draft_act_, n, c.ffn_size, stream_);
        ProjectDraft(L.ffn_down, draft_act_, n, draft_o_);
      } else {
        ProjectDraft(L.ffn_gate, draft_h_, n, draft_gate_);
        ProjectDraft(L.ffn_up, draft_h_, n, draft_up_);
        GeGlu(draft_gate_, draft_up_, draft_gate_, std::size_t{n} * c.ffn_size,
              stream_);
        ProjectDraft(L.ffn_down, draft_gate_, n, draft_o_);
      }
      const float* next = l + 1 < c.num_layers
                              ? dm.layers[l + 1].attn_norm.f32()
                              : dm.output_norm.f32();
      PostFeedForwardNorm(draft_o_, L.post_ffn_norm.f32(), L.output_scale,
                          draft_x_, next, draft_h_, n, hidden, eps, stream_);
    }
    // draft_h_ is the output-normalized state: vocabulary head and the
    // projection back to the target width for the next step. Proposals have
    // no decode twin to match, so the fastest one-row kernel reads the head
    // (repacked as Q4_K at load).
    // A lone chain on the small head scores the leading ordered rows, then
    // its table rows right after them.
    const auto format = GemvFormatOf(dm.token_embd.type);
    std::uint32_t scored = vocab;
    if (small &&
        LaunchKQuantGemv(*format, dm.token_embd.data, draft_h_, draft_logits_,
                         kDraftHeadRows, dm.token_embd.cols, stream_)) {
      if (!table.empty() &&
          !LaunchKQuantGemv(*format, jobs[0].cache->draft_table, draft_h_,
                            draft_logits_ + kDraftHeadRows,
                            static_cast<std::uint32_t>(table.size()),
                            dm.token_embd.cols, stream_)) {
        throw std::runtime_error("gemma4 draft table projection failed");
      }
      scored = kDraftHeadRows + static_cast<std::uint32_t>(table.size());
    } else if (n != 1 || !format ||
               !LaunchKQuantGemv(*format, dm.token_embd.data, draft_h_,
                                 draft_logits_, dm.token_embd.rows,
                                 dm.token_embd.cols, stream_)) {
      Project(dm.token_embd, draft_h_, nullptr, n, draft_logits_);
    }
    // The host picks each proposal (and where each chain ends) from its
    // row's top-64 candidates; the next step embeds them. One download and
    // one wait serve every row.
    const std::size_t count = std::min<std::size_t>(scored, kMtpCandidates);
    static_assert(offsetof(MtpCandidateLogits, logits) ==
                  kMtpCandidates * sizeof(std::uint32_t));
    for (std::uint32_t j = 0; j < n; ++j) {
      if (active[j] == 0) {
        continue;
      }
      std::uint32_t* ids = draft_candidates_ + j * candidate_ids;
      qwen38_flash_next::rocm::MtpTopCandidates(
          draft_logits_ + std::size_t{j} * vocab, ids,
          draft_candidate_scratch_ + j * candidate_ids,
          reinterpret_cast<float*>(ids + kMtpCandidates), scored, stream_);
      HIP_CHECK(hipMemcpyAsync(
          &draft_candidates_host_[j], ids,
          offsetof(MtpCandidateLogits, logits) + count * sizeof(float),
          hipMemcpyDeviceToHost, stream_));
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
    bool any = false;
    for (std::uint32_t j = 0; j < n; ++j) {
      if (active[j] == 0) {
        continue;
      }
      auto& host = draft_candidates_host_[j];
      host.size = count;
      // Head rows back to token ids.
      if (!head_tokens.empty()) {
        for (std::size_t k = 0; k < count; ++k) {
          const auto row = static_cast<std::uint32_t>(host.ids[k]);
          host.ids[k] = row >= kDraftHeadRows && scored != vocab
                            ? table[row - kDraftHeadRows]
                            : head_tokens[row];
        }
      }
      const DraftProposal proposal = jobs[j].propose(host);
      active[j] = 0;
      if (!proposal.token) {
        continue;
      }
      jobs[j].drafts->push_back(*proposal.token);
      tokens[j] = *proposal.token;
      if (!proposal.last && step + 1 < jobs[j].steps) {
        active[j] = 1;
        any = true;
      }
    }
    if (!any) {
      break;
    }
    HIP_CHECK(hipMemcpyAsync(draft_tokens_, tokens.data(),
                             n * sizeof(std::int32_t), hipMemcpyHostToDevice,
                             stream_));
    ProjectDraft(dm.post_projection, draft_h_, n, draft_next_);
  }
}

std::span<const std::uint32_t> Executor::ReadArgmax(std::size_t rows) const {
  if (rows == 0 || rows > kSplitRows) {
    throw std::invalid_argument("gemma4 argmax rows exceed capacity");
  }
  const std::size_t vocab = model_.vocab_size();
  const std::size_t scratch_floats =
      std::size_t{2} * kSplitRows * ((vocab + 1023) / 1024);
  hip::LaunchBatchedGPUArgmax(logits_, logit_index_, rows, vocab,
                              {gate_, scratch_floats}, stream_);
  HIP_CHECK(hipMemcpyAsync(argmax_host_, logit_index_,
                           rows * sizeof(std::uint32_t), hipMemcpyDeviceToHost,
                           stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
  return std::span<const std::uint32_t>(argmax_host_, rows);
}

void Executor::CommitLogits(KvCache& cache, std::uint32_t row) {
  BeforeMutation(cache);
  const std::size_t vocab = model_.vocab_size();
  HIP_CHECK(hipMemcpyAsync(cache.logits, logits_ + std::size_t{row} * vocab,
                           vocab * sizeof(float), hipMemcpyDeviceToDevice,
                           stream_));
}

void Executor::CopyCacheLogits(const KvCache& cache,
                               std::vector<float>* out) const {
  const std::size_t vocab = model_.vocab_size();
  HIP_CHECK(hipMemcpyAsync(logits_host_.data(), cache.logits,
                           vocab * sizeof(float), hipMemcpyDeviceToHost,
                           stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
  out->assign(logits_host_.begin(),
              logits_host_.begin() + static_cast<std::ptrdiff_t>(vocab));
}

std::span<const float> Executor::ReadLogits(std::size_t rows) const {
  const std::size_t count = rows * model_.vocab_size();
  if (count > logits_host_.size()) {
    throw std::invalid_argument("gemma4 logit read exceeds capacity");
  }
  if (count != 0) {
    HIP_CHECK(hipMemcpyAsync(logits_host_.data(), logits_,
                             count * sizeof(float), hipMemcpyDeviceToHost,
                             stream_));
  }
  HIP_CHECK(hipStreamSynchronize(stream_));
  return std::span<const float>(logits_host_.data(), count);
}

Executor::Candidates Executor::ReadCandidates(
    std::size_t rows, std::span<const std::uint32_t> extra) const {
  if (rows == 0 || rows > kSplitRows || extra.size() > kMaxCandidateExtra ||
      candidate_scratch_ == nullptr) {
    throw std::invalid_argument("gemma4 candidate rows exceed capacity");
  }
  const auto n_extra = static_cast<std::uint32_t>(extra.size());
  const std::uint32_t top = kCandidateTop + n_extra;
  if (n_extra != 0) {
    std::ranges::copy(extra, candidate_extra_host_.begin());
    HIP_CHECK(hipMemcpyAsync(candidate_extra_, candidate_extra_host_.data(),
                             n_extra * sizeof(std::uint32_t),
                             hipMemcpyHostToDevice, stream_));
  }
  auto* counts = reinterpret_cast<std::uint32_t*>(candidate_out_);
  auto* data = reinterpret_cast<sampling::LogitCandidate*>(
      candidate_out_ + kCandidateHeaderBytes);
  LogitCandidates(logits_, static_cast<std::uint32_t>(rows),
                  static_cast<std::uint32_t>(model_.vocab_size()), top,
                  candidate_extra_, n_extra, candidate_work_, counts, data,
                  stream_);
  const std::size_t stride = top + n_extra;
  HIP_CHECK(hipMemcpyAsync(
      candidate_host_.data(), candidate_out_,
      kCandidateHeaderBytes + rows * stride * sizeof(sampling::LogitCandidate),
      hipMemcpyDeviceToHost, stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
  const auto* host = candidate_host_.data();
  return {
      .counts = {reinterpret_cast<const std::uint32_t*>(host), rows},
      .data = {reinterpret_cast<const sampling::LogitCandidate*>(
                   host + kCandidateHeaderBytes),
               rows * stride},
      .top = top,
      .stride = static_cast<std::uint32_t>(stride),
  };
}

void Executor::CopyLogits(std::size_t rows, std::vector<float>* out) const {
  const auto span = ReadLogits(rows);
  out->assign(span.begin(), span.end());
}

}  // namespace gufo::models::gemma4::rocm
