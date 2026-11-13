#include "staging/models/glm5_next/kernels/rocm/executor.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "staging/models/glm5_next/kernels/rocm/attention.hpp"
#include "staging/models/glm5_next/kernels/rocm/dense_wmma.hpp"
#include "staging/models/glm5_next/kernels/rocm/gemv.hpp"
#include "staging/models/glm5_next/kernels/rocm/ops.hpp"
#include "staging/models/glm5_next/kernels/rocm/routed_compact.hpp"
#include "staging/models/glm5_next/kernels/rocm/routed_wmma.hpp"

namespace gufo::models::glm5_next::rocm {
namespace {

constexpr std::uint32_t kSelTile = 64;

bool TypeOf(core::GgmlType t, GemvType* out) {
  switch (t) {
    case core::GgmlType::kQ8_0:
      *out = GemvType::kQ8_0;
      return true;
    case core::GgmlType::kQ5_K:
      *out = GemvType::kQ5_K;
      return true;
    case core::GgmlType::kQ6_K:
      *out = GemvType::kQ6_K;
      return true;
    case core::GgmlType::kQ4_K:
      *out = GemvType::kQ4_K;
      return true;
    case core::GgmlType::kQ2_K:
      *out = GemvType::kQ2_K;
      return true;
    case core::GgmlType::kQ3_K:
      *out = GemvType::kQ3_K;
      return true;
    case core::GgmlType::kIQ2_XXS:
      *out = GemvType::kIQ2_XXS;
      return true;
    case core::GgmlType::kIQ1_M:
      *out = GemvType::kIQ1_M;
      return true;
    case core::GgmlType::kIQ3_XXS:
      *out = GemvType::kIQ3_XXS;
      return true;
    case core::GgmlType::kIQ4_XS:
      *out = GemvType::kIQ4_XS;
      return true;
    case core::GgmlType::kF32:
      *out = GemvType::kF32;
      return true;
    case core::GgmlType::kF16:
      *out = GemvType::kF16;
      return true;
    default:
      return false;
  }
}

// Stored type of a tensor validated by Executor::Create.
GemvType OfType(core::GgmlType t) {
  GemvType type = GemvType::kF32;
  const bool ok = TypeOf(t, &type);
  assert(ok);
  static_cast<void>(ok);
  return type;
}

// Expert stack types the bucketed tile tier launches (GemvRoutedTiles).
bool IsRoutedTier(GemvType type) {
  switch (type) {
    case GemvType::kIQ2_XXS:
    case GemvType::kIQ1_M:
    case GemvType::kIQ3_XXS:
    case GemvType::kIQ4_XS:
    case GemvType::kQ2_K:
    case GemvType::kQ3_K:
      return true;
    default:
      return false;
  }
}

bool Alloc(std::vector<void*>* pool, void** p, std::size_t bytes) {
  if (hipMalloc(p, bytes) != hipSuccess) {
    return false;
  }
  pool->push_back(*p);
  return true;
}

template <typename T>
bool AllocT(std::vector<void*>* pool, T** p, std::size_t count) {
  return Alloc(pool, reinterpret_cast<void**>(p), count * sizeof(T));
}

// Chunk-friendly GEMV dispatch: n == 1 stays on the plain tier, n > 1 walks
// the weights once for all rows through the Rows tier, whose per-row dots
// are bit-identical to the matching plain calls (and to each other across
// chunk widths, since each accumulator runs the same j-sequence). Dense
// Q5_K tiles of 16+ tokens take the WMMA tensor-core tier instead; its
// accumulation order differs from the scalar dot, so the quality contract
// lives in the dense_wmma test, not bit-equality.
void GemvN(std::uint32_t tokens, const DeviceTensor& t, const float* x,
           std::uint32_t x_stride, float* out, std::uint32_t out_stride) {
  if (tokens <= 1) {
    Gemv(t.data, OfType(t.type), t.rows, t.cols, t.row_bytes, x, out, nullptr);
    return;
  }
  assert(tokens <= Executor::kChunk);
  if (DenseWmmaSupports(OfType(t.type)) && tokens >= 16 && t.cols % 256 == 0U) {
    DenseWmma(t.data, OfType(t.type), t.rows, t.cols, x, x_stride, out,
              out_stride, tokens, nullptr);
    return;
  }
  GemvRows(t.data, OfType(t.type), tokens, t.rows, t.cols, t.row_bytes, x,
           x_stride, out, out_stride, nullptr);
}

}  // namespace

Session::~Session() {
  for (void* p : allocations_) {
    static_cast<void>(hipFree(p));
  }
}

void Session::Reset() {
  pos_ = -1;
  for (float* hist : conv_hist) {
    static_cast<void>(hipMemset(hist, 0, conv_bytes));
  }
  for (float* st : kda_state) {
    static_cast<void>(hipMemset(st, 0, state_bytes));
  }
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           std::string* error_msg) {
  const Config& c = model.config();
  const std::uint32_t h = c.hidden_size;
  const std::uint32_t hc = c.hc_count;
  const std::uint32_t inner = c.KdaInnerSize();

  // Validate every tensor the GEMV tier must decode (trunk and nextn).
  std::vector<std::reference_wrapper<const DeviceTensor>> tensors;
  tensors.push_back(model.token_embd());
  tensors.push_back(model.output());
  for (const DeviceLayer& l : model.layers()) {
    tensors.insert(tensors.end(),
                   {l.hc_attn_fn, l.hc_ffn_fn, l.attn_q_a, l.attn_q_b,
                    l.attn_kv_a, l.attn_k_b, l.attn_v_b, l.attn_out, l.ssm_q,
                    l.ssm_k, l.ssm_v, l.ssm_f_a, l.ssm_f_b, l.ssm_g_a,
                    l.ssm_g_b, l.ssm_beta, l.ssm_out, l.ffn_gate, l.ffn_up,
                    l.ffn_down, l.ffn_gate_exps, l.ffn_up_exps,
                    l.ffn_down_exps, l.shexp_gate, l.shexp_up, l.shexp_down,
                    l.idx_q_b, l.idx_k, l.idx_proj, l.idx_comp_gate});
  }
  for (const DeviceTensor& t : tensors) {
    if (t.empty()) {
      continue;
    }
    GemvType type;
    if (!TypeOf(t.type, &type)) {
      if (error_msg != nullptr) {
        *error_msg = "unsupported stored type for a resident tensor";
      }
      return nullptr;
    }
  }

  std::vector<void*> pool;
  std::unique_ptr<Executor> e(new Executor());
  e->model_ = &model;
  e->c_ = c;

  auto fail = [&](const std::string& msg) {
    if (error_msg != nullptr) {
      *error_msg = msg;
    }
    for (void* p : pool) {
      static_cast<void>(hipFree(p));
    }
    return nullptr;
  };

  const std::uint32_t heads = c.num_heads;
  const std::uint32_t nc = kChunk;
  const std::uint32_t iq_n = c.indexer_heads * c.indexer_key_len;
  const std::uint32_t max_sel = 4 * c.IndexerTopPools() + c.indexer_kpool - 1;
  const std::uint32_t tiles = (max_sel + kSelTile - 1) / kSelTile;
  const std::uint32_t f_w = std::max(c.dense_ff, c.shared_expert_ff);
  auto f = [&](float** p, std::size_t n) {
    return AllocT(&pool, p, n);
  };
  const std::size_t inner_n = static_cast<std::size_t>(nc) * inner;
  const std::size_t used_n = static_cast<std::size_t>(nc) * c.num_experts_used;
  // 16-slot bucket tiles: every bucket is a whole number of 16-row tiles
  // and the padded buckets occupy at most `slots + 15 * experts` rows.
  e->tiles_cap_ =
      static_cast<std::uint32_t>((used_n + 15U * c.num_experts + 15U) / 16U);
  // 32-slot tiles for the fused WMMA gate/up tier: the same bound with a
  // 32-row tile height.
  e->tiles_wide_cap_ = static_cast<std::uint32_t>(
      (used_n + 31U * c.num_experts + 31U) / 32U);
  if (!f(&e->w_, static_cast<std::size_t>(hc) * h) ||
      !f(&e->mean4_, hc) ||
      !f(&e->s0_, static_cast<std::size_t>(nc) * hc * h) ||
      !f(&e->s1_, static_cast<std::size_t>(nc) * hc * h) ||
      !f(&e->x_, static_cast<std::size_t>(nc) * h) ||
      !f(&e->y0_, static_cast<std::size_t>(nc) * h) ||
      !f(&e->mix_a_, static_cast<std::size_t>(nc) * hc * (2 + hc)) ||
      !f(&e->pre_a_, static_cast<std::size_t>(nc) * hc) ||
      !f(&e->post_a_, static_cast<std::size_t>(nc) * hc) ||
      !f(&e->comb_a_, static_cast<std::size_t>(nc) * hc * hc) ||
      !f(&e->q_, static_cast<std::size_t>(nc) *
                       std::max<std::size_t>(c.MlaQSize(), inner)) ||
      !f(&e->head_out_, static_cast<std::size_t>(nc) * c.MlaVSize()) ||
      !f(&e->latent_, static_cast<std::size_t>(nc) * hc * h) ||
      !f(&e->dsa_latent_, static_cast<std::size_t>(nc) * heads *
                              c.kv_lora_rank) ||
      !f(&e->q_abs_, static_cast<std::size_t>(nc) * heads *
                         c.kv_lora_rank) ||
      !f(&e->kv_, static_cast<std::size_t>(nc) * c.kv_lora_rank) ||
      !f(&e->qr_, static_cast<std::size_t>(nc) * c.q_lora_rank) ||
      !f(&e->iq_, static_cast<std::size_t>(nc) * iq_n) ||
      !f(&e->iw_, static_cast<std::size_t>(nc) * c.indexer_heads) ||
      !f(&e->vq_, inner_n) || !f(&e->vk_, inner_n) || !f(&e->eg_, inner_n) ||
      !f(&e->sk_, inner_n) || !f(&e->ko_, inner_n) || !f(&e->g2_, inner_n) ||
      !f(&e->g1_, inner_n) ||
      !f(&e->beta_, static_cast<std::size_t>(nc) * c.kda_heads) ||
      !f(&e->moe_w_, used_n) ||
       !f(&e->tile_scores_, static_cast<std::size_t>(nc) * tiles * heads *
                                kSelTile) ||
      !f(&e->moe_gate_, used_n * c.expert_ff) ||
      !f(&e->moe_up_, used_n * c.expert_ff) ||
      !f(&e->moe_inter_, used_n * c.expert_ff) ||
      !f(&e->moe_down_, used_n * h) ||
      !f(&e->f_gate_, static_cast<std::size_t>(nc) * f_w) ||
      !f(&e->f_up_, static_cast<std::size_t>(nc) * f_w) ||
      !f(&e->f_inter_, static_cast<std::size_t>(nc) * f_w) ||
      !f(&e->router_, static_cast<std::size_t>(nc) * c.num_experts) ||
      !f(&e->h_, static_cast<std::size_t>(nc) * h) ||
      !f(&e->logits_, c.vocab_size) ||
      !AllocT(&pool, &e->overflow_, 1) ||
      !AllocT(&pool, &e->ids_, used_n) ||
      !AllocT(&pool, &e->expert_counts_, c.num_experts) ||
      !AllocT(&pool, &e->pad_bounds_, c.num_experts + 1) ||
      !AllocT(&pool, &e->cursors_, c.num_experts) ||
      !AllocT(&pool, &e->rows_slot_,
              RoutedCompactRows(used_n, c.num_experts)) ||
      !AllocT(&pool, &e->tiles_, e->tiles_cap_) ||
      !AllocT(&pool, &e->tiles_wide_, e->tiles_wide_cap_) ||
      !AllocT(&pool, &e->x_half_, static_cast<std::size_t>(nc) * h) ||
      !AllocT(&pool, &e->moe_inter_half_, used_n * c.expert_ff) ||
      !AllocT(&pool, &e->head_ids_, static_cast<std::size_t>(nc) * heads)) {
    return fail("Executor scratch allocation failed");
  }

  std::vector<float> ones(static_cast<std::size_t>(hc) * h, 1.0F);
  std::vector<float> mean(hc, 1.0F / static_cast<float>(hc));
  std::vector<std::int32_t> head_ids(static_cast<std::size_t>(nc) * heads);
  for (std::uint32_t t = 0; t < nc; ++t) {
    for (std::uint32_t hd = 0; hd < heads; ++hd) {
      head_ids[static_cast<std::size_t>(t) * heads + hd] =
          static_cast<std::int32_t>(hd);
    }
  }
  if (hipMemcpy(e->w_, ones.data(), ones.size() * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(e->mean4_, mean.data(), sizeof(float) * hc,
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(e->head_ids_, head_ids.data(),
                head_ids.size() * sizeof(std::int32_t),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemset(e->overflow_, 0, sizeof(std::uint32_t)) != hipSuccess) {
    return fail("Executor constant upload failed");
  }
  return e;
}

std::size_t Executor::SessionBytes(std::uint32_t capacity) const {
  const Config& c = c_;
  std::size_t dsa = 0;
  std::size_t kda = 0;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (c.IsDsaLayer(il)) {
      ++dsa;
    } else {
      ++kda;
    }
  }
  const std::size_t latent = sizeof(__half) * c.kv_lora_rank;
  const std::size_t pool = sizeof(float) * 2 * c.indexer_key_len;
  const std::size_t conv = sizeof(float) * 3 * 3 * c.KdaInnerSize();
  const std::size_t state = sizeof(float) * c.kda_heads * c.kda_head_dim *
                            c.kda_head_dim;
  const std::size_t n_pools =
      static_cast<std::size_t>(capacity) / c.indexer_kpool + 1;
  const std::size_t np_stride = n_pools + 1;
  const std::size_t sel_stride =
      4ULL * c.IndexerTopPools() + c.indexer_kpool;
  return dsa * capacity * (latent + pool) + kda * (conv + state) +
         kChunk * np_stride * sizeof(float) +
         kChunk * static_cast<std::size_t>(c.IndexerTopPools()) *
             sizeof(std::int32_t) +
         kChunk * sel_stride * sizeof(std::int32_t) +
         (n_pools + 4096) * sizeof(std::uint64_t);
}

std::unique_ptr<Session> Executor::CreateSession(
    std::uint32_t capacity, std::string* error_msg) const {
  const Config& c = c_;
  auto s = std::unique_ptr<Session>(new Session());
  s->cap_ = capacity;
  std::vector<void*>* pool = &s->allocations_;

  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (c.IsDsaLayer(il)) {
      __half* lat = nullptr;
      float* rows = nullptr;
      if (!AllocT(pool, &lat,
                  static_cast<std::size_t>(capacity) * c.kv_lora_rank) ||
          !AllocT(pool, &rows, static_cast<std::size_t>(capacity) * 2 *
                                   c.indexer_key_len)) {
        if (error_msg != nullptr) {
          *error_msg = "session cache allocation failed";
        }
        return nullptr;
      }
      s->latent_cache.push_back(lat);
      s->pool_cache.push_back(rows);
    } else {
      const std::size_t conv = 3ULL * 3 * c.KdaInnerSize();
      const std::size_t state = static_cast<std::size_t>(c.kda_heads) *
                                c.kda_head_dim * c.kda_head_dim;
      float* hist = nullptr;
      float* st = nullptr;
      if (!AllocT(pool, &hist, conv) || !AllocT(pool, &st, state)) {
        if (error_msg != nullptr) {
          *error_msg = "session state allocation failed";
        }
        return nullptr;
      }
      static_cast<void>(hipMemset(hist, 0, conv * sizeof(float)));
      static_cast<void>(hipMemset(st, 0, state * sizeof(float)));
      s->conv_bytes = conv * sizeof(float);
      s->state_bytes = state * sizeof(float);
      s->conv_hist.push_back(hist);
      s->kda_state.push_back(st);
    }
  }
  const std::size_t n_pools =
      static_cast<std::size_t>(capacity) / c.indexer_kpool + 1;
  s->pool_scores_stride = n_pools + 1;
  s->pool_sel_stride = c.IndexerTopPools();
  s->sel_stride = 4ULL * c.IndexerTopPools() + c.indexer_kpool;
  if (!AllocT(pool, &s->pool_scores, kChunk * s->pool_scores_stride) ||
      !AllocT(pool, &s->pool_sel, kChunk * s->pool_sel_stride) ||
      !AllocT(pool, &s->sel, kChunk * s->sel_stride) ||
      !AllocT(pool, &s->topk_scratch, n_pools + 4096)) {
    if (error_msg != nullptr) {
      *error_msg = "session scratch allocation failed";
    }
    return nullptr;
  }
  return s;
}

void Executor::Dbg(const char* fmt, std::uint32_t il, const float* dev,
                   std::size_t n) const {
  if (dbg_dir_.empty()) {
    return;
  }
  std::vector<float> host(n);
  if (dev == nullptr ||
      hipMemcpy(host.data(), dev, n * sizeof(float), hipMemcpyDeviceToHost) !=
          hipSuccess) {
    return;
  }
  char tag[96];
  char path[640];
  std::snprintf(tag, sizeof(tag), fmt, il);
  std::snprintf(path, sizeof(path), "%s/%s.f32", dbg_dir_.c_str(), tag);
  FILE* fp = std::fopen(path, "wb");
  if (fp == nullptr) {
    return;
  }
  std::fwrite(host.data(), sizeof(float), n, fp);
  std::fclose(fp);
}

void Executor::KdaLayer(const DeviceLayer& l, std::uint32_t slot,
                        const float* x, std::uint32_t n, std::int64_t p0,
                        Session& s, float* y) {
  const Config& c = c_;
  const std::uint32_t heads = c.kda_heads;
  const std::uint32_t d = c.kda_head_dim;
  const std::uint32_t inner = c.KdaInnerSize();
  const std::size_t kv_p = c.kv_lora_rank;
  float* hist = s.conv_hist[slot];

  // Projections for the whole chunk (raw k rows live in sk_, raw v rows in
  // eg_; both regions are consumed by their token loops in order before the
  // later per-token reuse writes them again).
  GemvN(n, l.ssm_q, x, c.hidden_size, vq_, inner);
  GemvN(n, l.ssm_k, x, c.hidden_size, sk_, inner);
  GemvN(n, l.ssm_v, x, c.hidden_size, eg_, inner);
  GemvN(n, l.ssm_beta, x, c.hidden_size, beta_, heads);
  SigmoidInPlace(beta_, n * heads, nullptr);
  GemvN(n, l.ssm_f_a, x, c.hidden_size, kv_, kv_p);
  GemvN(n, l.ssm_f_b, kv_, kv_p, g1_, inner);
  AddBias(g1_, l.ssm_dt.f32(), n * inner, inner, nullptr);

  // Convolutions and the recurrence are strictly sequential over positions
  // and run as one launch per stream / stage: conv q ends in ko_, conv k in
  // q_, v in vk_ (one row per token).
  KdaConvStepChunk(static_cast<std::uint64_t>(p0), vq_,
                   l.ssm_conv_q.f32(), hist, inner, n, ko_, nullptr);
  KdaConvStepChunk(static_cast<std::uint64_t>(p0), sk_, l.ssm_conv_k.f32(),
                   hist + 3 * inner, inner, n, q_, nullptr);
  KdaConvStepChunk(static_cast<std::uint64_t>(p0), eg_, l.ssm_conv_v.f32(),
                   hist + 6 * inner, inner, n, vk_, nullptr);

  // Forget gate eg = exp(lb * sigmoid(-(g1 * a))) now that the raw v rows
  // of eg_ have been consumed by the convolutions.
  KdaEGate(g1_, l.ssm_a.f32(), heads, d, c.kda_gate_lower_bound, eg_, n,
           nullptr);

  // FLA l2 norms; the recurrent applies 1/sqrt(d) to the query only
  // (build_delta_net_autoregressive scales q after both norms). Fold and
  // update run in one chunk launch: the state is shared, so every position
  // must decay and update before the next one folds.
  L2NormHeads(q_, heads * n, d, c.ln_eps, nullptr);
  L2NormHeads(ko_, heads * n, d, c.ln_eps, nullptr);
  Scale(ko_, inner * n, 1.0F / std::sqrt(static_cast<float>(d)), nullptr);
  if (!KdaFoldUpdateChunk(s.kda_state[slot], eg_, q_, ko_, vk_, beta_, heads,
                          d, n, g2_, nullptr)) {
    float* sk_scratch = sk_;  // raw k rows are fully consumed by the convs
    for (std::uint32_t r = 0; r < n; ++r) {
      KdaFoldSk(s.kda_state[slot], eg_ + static_cast<std::size_t>(r) * inner,
                q_ + static_cast<std::size_t>(r) * inner, heads, d,
                sk_scratch + static_cast<std::size_t>(r) * inner, nullptr);
      KdaUpdateOut(s.kda_state[slot], q_ + static_cast<std::size_t>(r) * inner,
                   ko_ + static_cast<std::size_t>(r) * inner,
                   vk_ + static_cast<std::size_t>(r) * inner,
                   beta_ + static_cast<std::size_t>(r) * heads,
                   sk_scratch + static_cast<std::size_t>(r) * inner, heads, d,
                   g2_ + static_cast<std::size_t>(r) * inner, nullptr);
    }
  }

  // Output gate: rms_norm(o, ssm_norm) * sigmoid(g_b(g_a(x))).
  GemvN(n, l.ssm_g_a, x, c.hidden_size, kv_, kv_p);
  GemvN(n, l.ssm_g_b, kv_, kv_p, g1_, inner);
  KdaOutGate(g2_, l.ssm_norm.f32(), g1_, ko_, heads * n, d, c.rms_eps,
             nullptr);
  GemvN(n, l.ssm_out, ko_, inner, y, c.hidden_size);

  if (n == 1) {
    Dbg("kda_q_gemm-%u", dbg_il_, vq_, inner);
    Dbg("kda_q_conv-%u", dbg_il_, ko_, inner);
    Dbg("kda_k_gemm-%u", dbg_il_, sk_, inner);
    Dbg("kda_k_conv-%u", dbg_il_, q_, inner);
    Dbg("kda_v_gemm-%u", dbg_il_, eg_, inner);
    Dbg("kda_v_conv-%u", dbg_il_, vk_, inner);
    Dbg("kda_beta-%u", dbg_il_, beta_, heads);
    Dbg("kda_fa-%u", dbg_il_, kv_, d);
    Dbg("kda_g1-%u", dbg_il_, g1_, inner);
    Dbg("kda_dt_bias-%u", dbg_il_, l.ssm_dt.f32(), inner);
    Dbg("kda_scan_out-%u", dbg_il_, g2_, inner);
    Dbg("kda_g2-%u", dbg_il_, g1_, inner);
    Dbg("kda_normed-%u", dbg_il_, ko_, inner);
    Dbg("kda_out-%u", dbg_il_, y, c.hidden_size);
  }
}

void Executor::DsaLayer(const DeviceLayer& l, std::uint32_t slot,
                        const float* x, std::uint32_t n, std::int64_t p0,
                        Session& s, float* y) {
  const Config& c = c_;
  const std::uint32_t heads = c.num_heads;
  const std::uint32_t key = c.indexer_key_len;
  const std::uint32_t kpool = c.indexer_kpool;
  const std::uint32_t latent = c.kv_lora_rank;
  const std::size_t kv_p = latent;
  const std::size_t iq_n = static_cast<std::size_t>(c.indexer_heads) * key;
  float* pool = s.pool_cache[slot];
  __half* cache = s.latent_cache[slot];

  // Chunk prologue: every cache row and projection of the chunk is written
  // up front. A token's selection only scores complete pools and tail slots
  // up to its own position, and the gathered attention only reads cache rows
  // <= that position, so writing the later rows first changes nothing.
  // Indexer inputs: LayerNorm(attn_k) into the pool row, then the raw gate
  // straight into its [ik | gate] gap (pitch 2 * key).
  GemvN(n, l.idx_k, x, c.hidden_size, kv_, kv_p);
  for (std::uint32_t r = 0; r < n; ++r) {
    LayerNorm(kv_ + static_cast<std::size_t>(r) * kv_p, l.idx_k_norm.f32(),
              l.idx_k_bias.f32(),
              pool + (static_cast<std::size_t>(p0) + r) * 2 * key, key,
              c.ln_eps, nullptr);
  }
  GemvN(n, l.idx_comp_gate, x, c.hidden_size, pool + p0 * 2 * key + key,
        2 * key);

  // Absorbed MLA: the normed latent rows join the F16 cache; the queries
  // walk the lora, the per-head kb absorption and the indexer projections.
  GemvN(n, l.attn_kv_a, x, c.hidden_size, kv_, kv_p);
  RmsNorm(kv_, l.attn_kv_a_norm.f32(), kv_, n, latent, kv_p, kv_p, c.rms_eps,
          nullptr);
  for (std::uint32_t r = 0; r < n; ++r) {
    ConvertF32ToHalf(kv_ + static_cast<std::size_t>(r) * kv_p,
                     cache + (static_cast<std::size_t>(p0) + r) * latent,
                     latent, overflow_, nullptr);
  }
  GemvN(n, l.attn_q_a, x, c.hidden_size, qr_, c.q_lora_rank);
  RmsNorm(qr_, l.attn_q_a_norm.f32(), qr_, n, c.q_lora_rank, c.q_lora_rank,
          c.q_lora_rank, c.rms_eps, nullptr);
  GemvN(n, l.attn_q_b, qr_, c.q_lora_rank, q_, c.MlaQSize());
  GemvN(n, l.idx_q_b, qr_, c.q_lora_rank, iq_, iq_n);
  GemvN(n, l.idx_proj, x, c.hidden_size, iw_, c.indexer_heads);
  // Per-head k absorption runs one plain GEMV per head over the stacked
  // Expert() slice (Q8_0 heads have no grouped kernel); the multi-token
  // pass stacks all heads into one grouped launch. Slots are the
  // (token, head) pairs: token r sits at x + r * x_stride.
  if (n == 1) {
    for (std::uint32_t hd = 0; hd < heads; ++hd) {
      Gemv(l.attn_k_b.Expert(hd), OfType(l.attn_k_b.type), l.attn_k_b.rows,
           l.attn_k_b.cols, l.attn_k_b.row_bytes, q_ + hd * l.attn_k_b.cols,
           q_abs_ + hd * latent, nullptr);
    }
  } else {
    assert(l.attn_k_b.rows == latent);
    GemvRowsGrouped(l.attn_k_b.Expert(0), OfType(l.attn_k_b.type), n, heads,
                    l.attn_k_b.rows, l.attn_k_b.cols, l.attn_k_b.row_bytes, q_,
                    c.MlaQSize(), l.attn_k_b.cols, q_abs_, heads * latent,
                    nullptr);
  }
  // The reference softmax scale is 1/sqrt(n_embd_head_k_mla) (llama-graph
  // build_attn_mha kq_scale, glm5-next.cpp); folding it into the absorbed q
  // rows is the same temperature applied before the latent dot product.
  Scale(q_abs_, n * heads * latent,
        1.0F / std::sqrt(static_cast<float>(c.mla_head_dim)), nullptr);

  // Per token: pool scores over the complete pools only (the current
  // partial pool joins through the tail); stable top-k; sel expansion;
  // gather softmax over the selected latent rows. One token runs the trio
  // on the shared selection scratch; a prefill chunk replaces the
  // per-token launches with one token-batched launch per kernel whose
  // blocks run the identical arithmetic (_Batch contracts), while the
  // top-k stays a per-token loop until its own batched form lands.
  const std::uint32_t top_pools = c.IndexerTopPools();
  const std::uint32_t use_tail = c.kpool_select_tail ? 1U : 0U;
  if (n == 1) {
    const std::int32_t pos = static_cast<std::int32_t>(p0);
    const std::uint32_t tokens = static_cast<std::uint32_t>(pos) + 1;
    const std::uint32_t n_pools = tokens / kpool;
    std::uint32_t k = 0;
    if (n_pools > 0) {
      IdxScore(pool, 2 * key, n_pools * kpool, kpool, key, c.indexer_heads,
               iq_, l.idx_comp_ape.f32(), iw_,
               1.0F / std::sqrt(static_cast<float>(key) * c.indexer_heads),
               s.pool_scores, nullptr);
      k = std::min(n_pools, top_pools);
      TopKDesc(s.pool_scores, n_pools, k, s.pool_sel, s.topk_scratch,
               nullptr);
    }
    const std::uint32_t n_tail = c.kpool_select_tail ? tokens % kpool : 0;
    BuildSel(s.pool_sel, k, pos, n_tail, s.sel, nullptr);
    const std::uint32_t n_sel = k * kpool + n_tail;
    MlaGatherScore(q_abs_, cache, s.sel, n_sel, kSelTile, heads, tile_scores_,
                   nullptr);
    MlaGatherCombine(tile_scores_, cache, s.sel, kSelTile, n_sel, heads,
                     dsa_latent_, nullptr);
  } else {
    IdxScoreBatch(pool, 2 * key, static_cast<std::uint32_t>(p0) + 1U, kpool,
                  key, c.indexer_heads, iq_, iq_n, l.idx_comp_ape.f32(), iw_,
                  c.indexer_heads,
                  1.0F / std::sqrt(static_cast<float>(key) * c.indexer_heads),
                  n, s.pool_scores, s.pool_scores_stride, nullptr);
    for (std::uint32_t r = 0; r < n; ++r) {
      const std::uint32_t n_pools =
          (static_cast<std::uint32_t>(p0) + r + 1U) / kpool;
      const std::uint32_t k = std::min(n_pools, top_pools);
      if (k > 0) {
        TopKDesc(s.pool_scores +
                     static_cast<std::size_t>(r) * s.pool_scores_stride,
                 n_pools, k,
                 s.pool_sel + static_cast<std::size_t>(r) * s.pool_sel_stride,
                 s.topk_scratch, nullptr);
      }
    }
    const std::uint32_t sel_capacity = 4U * top_pools + kpool - 1U;
    BuildSelBatch(s.pool_sel, s.pool_sel_stride,
                  static_cast<std::uint32_t>(p0), top_pools, kpool, use_tail,
                  n, sel_capacity, s.sel, s.sel_stride, nullptr);
    const std::size_t scores_stride =
        static_cast<std::size_t>((sel_capacity + kSelTile - 1U) / kSelTile) *
        heads * kSelTile;
    const std::size_t qa_stride = static_cast<std::size_t>(heads) * latent;
    const std::uint32_t tokens_last = static_cast<std::uint32_t>(p0) + n;
    const std::uint32_t n_sel_max =
        std::min(tokens_last / kpool, top_pools) * kpool +
        (c.kpool_select_tail ? tokens_last % kpool : 0U);
    MlaGatherScoreBatch(q_abs_, qa_stride, cache, s.sel, s.sel_stride,
                        n_sel_max, static_cast<std::uint32_t>(p0), kpool,
                        top_pools, use_tail, n, kSelTile, heads,
                        tile_scores_, scores_stride, nullptr);
    MlaGatherCombineBatch(tile_scores_, scores_stride, cache, s.sel,
                          s.sel_stride, n_sel_max,
                          static_cast<std::uint32_t>(p0), kpool, top_pools,
                          use_tail, n, kSelTile, heads, dsa_latent_,
                          qa_stride, nullptr);
  }

  // Per-head value maps (same head loop shape as the k absorption, same
  // grouped multi-token launch) and the chunked output projection.
  if (n == 1) {
    for (std::uint32_t hd = 0; hd < heads; ++hd) {
      Gemv(l.attn_v_b.Expert(hd), OfType(l.attn_v_b.type), l.attn_v_b.rows,
           l.attn_v_b.cols, l.attn_v_b.row_bytes, dsa_latent_ + hd * latent,
           head_out_ + hd * l.attn_v_b.rows, nullptr);
    }
  } else {
    assert(heads * l.attn_v_b.rows == c.MlaVSize());
    GemvRowsGrouped(l.attn_v_b.Expert(0), OfType(l.attn_v_b.type), n, heads,
                    l.attn_v_b.rows, l.attn_v_b.cols, l.attn_v_b.row_bytes,
                    dsa_latent_, heads * latent, latent, head_out_,
                    c.MlaVSize(), nullptr);
  }
  GemvN(n, l.attn_out, head_out_, c.MlaVSize(), y, c.hidden_size);

  if (n == 1) {
    Dbg("indexer_k-%u", dbg_il_, pool + p0 * 2 * key, key);
    Dbg("kv_cmpr-%u", dbg_il_, kv_, latent);
    Dbg("q_resid-%u", dbg_il_, qr_, c.q_lora_rank);
    Dbg("indexer_q-%u", dbg_il_, iq_, c.indexer_heads * key);
    Dbg("indexer_gate-%u", dbg_il_, iw_, c.indexer_heads);
    Dbg("q_absorbed-%u", dbg_il_, q_abs_, heads * latent);
    Dbg("kqv_out-%u", dbg_il_, head_out_, c.MlaVSize());
    Dbg("attn_out-%u", dbg_il_, y, c.hidden_size);
  }
}

void Executor::Ffn(const DeviceLayer& l, std::uint32_t il, const float* x,
                   std::uint32_t n, float* y) {
  const Config& c = c_;
  const std::uint32_t h = c.hidden_size;
  if (c.IsDenseFfnLayer(il)) {
    GemvN(n, l.ffn_gate, x, h, f_gate_, c.dense_ff);
    GemvN(n, l.ffn_up, x, h, f_up_, c.dense_ff);
    const float clamp = c.SwigluClampShexp(il);
    SwigluClamp(f_gate_, f_up_, f_inter_, n, c.dense_ff, c.dense_ff,
                c.dense_ff, c.dense_ff, clamp, clamp, nullptr);
    GemvN(n, l.ffn_down, f_inter_, c.dense_ff, y, h);
    if (n == 1) {
      Dbg("ffn_out-%u", il, y, h);
    }
    return;
  }

  const std::uint32_t used = c.num_experts_used;
  GemvN(n, l.router, x, h, router_, c.num_experts);
  MoERoute(router_, l.exp_probs_b.f32(), c.num_experts, used,
           c.expert_weights_scale, moe_w_, ids_, n, nullptr);

  // One sweep over all N * used expert slots, weight row per tile. Below
  // `kBucketMinRows` the grouped sweep follows the slot order directly;
  // from there the slots are compacted into 16-row padded expert buckets
  // and each decode of a weight row serves up to 16 slot rows of the same
  // expert. The gate/up gemms share the hidden-state row of each token
  // (slots_per_row = used); the down gemm consumes the own activation row
  // of every slot (slots_per_row = 1). Per-slot dots and stores are the
  // same order as the grouped tier, so the outputs are bit-identical.
  const std::uint32_t slots = n * used;
  const bool bucketed = n >= kBucketMinRows &&
                        IsRoutedTier(OfType(l.ffn_gate_exps.type)) &&
                        IsRoutedTier(OfType(l.ffn_up_exps.type)) &&
                        IsRoutedTier(OfType(l.ffn_down_exps.type));
  // Over low-bit IQ stacks the wide chunk additionally takes the WMMA
  // tensor-core tier: one launch dots the 32-slot expert tiles against an
  // F16 copy of the activations. IQ1_M/IQ2_XXS gate/up replace both gate/up
  // GEMVs and the clamp pass (fused swiglu); IQ3_XXS down replaces the down
  // GEMV on the F16 copy of moe_inter. Tensor-core accumulation order
  // differs from the scalar dot, so tier quality is held by the routed_wmma
  // contract, not bit-equality.
  const GemvType gu_type = OfType(l.ffn_gate_exps.type);
  const bool gu_wmma =
      bucketed && gu_type == OfType(l.ffn_up_exps.type) &&
      (gu_type == GemvType::kIQ1_M || gu_type == GemvType::kIQ2_XXS);
  const bool down_wmma =
      bucketed && OfType(l.ffn_down_exps.type) == GemvType::kIQ3_XXS;
  const bool tier_wide = gu_wmma || down_wmma;
  const float clamp = c.SwigluClampExp(il);
  if (bucketed) {
    ExpertCounts(ids_, slots, c.num_experts, expert_counts_, nullptr);
    BuildRoutedTiles(expert_counts_, c.num_experts, 16U, tiles_cap_, tiles_,
                     tier_wide ? 32U : 16U, tiles_wide_cap_,
                     tier_wide ? tiles_wide_ : tiles_, nullptr);
    RoutedCompact(ids_, expert_counts_, pad_bounds_, cursors_, rows_slot_, n,
                  used, c.num_experts, nullptr);
  }
  if (gu_wmma) {
    ConvertF32ToHalf(x, x_half_, static_cast<std::size_t>(n) * h, overflow_,
                     nullptr);
    RoutedGateUpWmma(gu_type, l.ffn_gate_exps.data, l.ffn_up_exps.data,
                     l.ffn_gate_exps.row_bytes * l.ffn_gate_exps.rows,
                     static_cast<std::size_t>(c.num_experts) *
                         l.ffn_gate_exps.row_bytes * l.ffn_gate_exps.rows,
                     x_half_, h, tiles_wide_, tiles_wide_cap_, pad_bounds_,
                     rows_slot_, used, l.ffn_gate_exps.rows,
                     l.ffn_gate_exps.cols, clamp, moe_inter_, nullptr);
  } else if (bucketed) {
    GemvRoutedTiles(l.ffn_gate_exps.data, OfType(l.ffn_gate_exps.type),
                    l.ffn_gate_exps.row_bytes * l.ffn_gate_exps.rows, tiles_,
                    tiles_cap_, pad_bounds_, rows_slot_, used,
                    l.ffn_gate_exps.rows, l.ffn_gate_exps.cols, x, h, moe_gate_,
                    nullptr);
    GemvRoutedTiles(l.ffn_up_exps.data, OfType(l.ffn_up_exps.type),
                    l.ffn_up_exps.row_bytes * l.ffn_up_exps.rows, tiles_,
                    tiles_cap_, pad_bounds_, rows_slot_, used,
                    l.ffn_up_exps.rows, l.ffn_up_exps.cols, x, h, moe_up_,
                    nullptr);
  } else {
    GemvGrouped(l.ffn_gate_exps.data, OfType(l.ffn_gate_exps.type),
                l.ffn_gate_exps.row_bytes * l.ffn_gate_exps.rows, ids_,
                n * used, used, l.ffn_gate_exps.rows, l.ffn_gate_exps.cols, x,
                h, moe_gate_, nullptr);
    GemvGrouped(l.ffn_up_exps.data, OfType(l.ffn_up_exps.type),
                l.ffn_up_exps.row_bytes * l.ffn_up_exps.rows, ids_, n * used,
                used, l.ffn_up_exps.rows, l.ffn_up_exps.cols, x, h, moe_up_,
                nullptr);
  }
  if (!gu_wmma) {
    SwigluClamp(moe_gate_, moe_up_, moe_inter_, n * used, c.expert_ff,
                c.expert_ff, c.expert_ff, c.expert_ff, clamp, clamp, nullptr);
  }
  if (down_wmma) {
    ConvertF32ToHalf(moe_inter_, moe_inter_half_,
                     static_cast<std::size_t>(slots) * c.expert_ff, overflow_,
                     nullptr);
    RoutedDownWmma(OfType(l.ffn_down_exps.type), l.ffn_down_exps.data,
                   l.ffn_down_exps.row_bytes * l.ffn_down_exps.rows,
                   static_cast<std::size_t>(c.num_experts) *
                       l.ffn_down_exps.row_bytes * l.ffn_down_exps.rows,
                   moe_inter_half_, c.expert_ff, tiles_wide_, tiles_wide_cap_,
                   pad_bounds_, rows_slot_, l.ffn_down_exps.rows,
                   l.ffn_down_exps.cols, moe_down_, nullptr);
  } else if (bucketed) {
    GemvRoutedTiles(l.ffn_down_exps.data, OfType(l.ffn_down_exps.type),
                    l.ffn_down_exps.row_bytes * l.ffn_down_exps.rows, tiles_,
                    tiles_cap_, pad_bounds_, rows_slot_, 1U,
                    l.ffn_down_exps.rows, l.ffn_down_exps.cols, moe_inter_,
                    c.expert_ff, moe_down_, nullptr);
  } else {
    GemvGrouped(l.ffn_down_exps.data, OfType(l.ffn_down_exps.type),
                l.ffn_down_exps.row_bytes * l.ffn_down_exps.rows, ids_,
                n * used, 1U, l.ffn_down_exps.rows, l.ffn_down_exps.cols,
                moe_inter_, c.expert_ff, moe_down_, nullptr);
  }

  // Shared expert into y, then the weighted routed experts fold in.
  GemvN(n, l.shexp_gate, x, h, f_gate_, c.shared_expert_ff);
  GemvN(n, l.shexp_up, x, h, f_up_, c.shared_expert_ff);
  const float sh_clamp = c.SwigluClampShexp(il);
  SwigluClamp(f_gate_, f_up_, f_inter_, n, c.shared_expert_ff,
              c.shared_expert_ff, c.shared_expert_ff, c.shared_expert_ff,
              sh_clamp, sh_clamp, nullptr);
  GemvN(n, l.shexp_down, f_inter_, c.shared_expert_ff, y, h);
  WeightedSumAdd(y, moe_down_, moe_w_, used, h, h, n, nullptr);

  if (n == 1) {
    Dbg("moe_logits-%u", il, router_, c.num_experts);
    Dbg("moe_w-%u", il, moe_w_, used);
    if (hipMemcpy(router_, ids_, used * sizeof(std::int32_t),
                  hipMemcpyDeviceToDevice) == hipSuccess) {
      Dbg("moe_ids-%u", il, router_, used);  // int32 bit patterns
    }
    Dbg("moe_gate-%u", il, moe_gate_,
        static_cast<std::size_t>(used) * c.expert_ff);
    Dbg("moe_inter-%u", il, moe_inter_,
        static_cast<std::size_t>(used) * c.expert_ff);
    Dbg("moe_down-%u", il, moe_down_, static_cast<std::size_t>(used) * h);
    Dbg("ffn_shexp-%u", il, y, h);
    Dbg("ffn_out-%u", il, y, h);
  }
}

void Executor::Step(Session& session, std::int32_t token) {
  RunTokens(session, &token, 1U);
}

void Executor::Prefill(Session& session, const std::int32_t* tokens,
                       std::size_t count) {
  // The anchor dump keeps one file per value, so a debug run stays on the
  // one-token path and preserves per-step semantics.
  const std::size_t step = dbg_dir_.empty() ? kChunk : 1U;
  for (std::size_t done = 0; done < count; done += step) {
    RunTokens(session, tokens + done,
              static_cast<std::uint32_t>(std::min(step, count - done)));
  }
}

void Executor::RunTokens(Session& session, const std::int32_t* tokens,
                         std::uint32_t n) {
  const Config& c = c_;
  const std::uint32_t h = c.hidden_size;
  const std::uint32_t hc = c.hc_count;
  assert(n >= 1U && n <= kChunk);
  assert(session.pos_ + static_cast<std::int64_t>(n) <
         static_cast<std::int64_t>(session.cap_));
  const std::int64_t p0 = ++session.pos_;
  const std::size_t sp = static_cast<std::size_t>(hc) * h;
  const std::size_t mix_w = hc * (2 + hc);

  for (std::uint32_t r = 0; r < n; ++r) {
    float* emb = s0_ + static_cast<std::size_t>(r) * sp;
    EmbedRow(model_->token_embd().data, OfType(model_->token_embd().type),
             static_cast<std::uint32_t>(tokens[r]), h, emb, nullptr);
    for (std::uint32_t i = 1; i < hc; ++i) {
      static_cast<void>(hipMemcpyAsync(
          emb + static_cast<std::size_t>(i) * h, emb, sizeof(float) * h,
          hipMemcpyDeviceToDevice, nullptr));
    }
  }

  // latent_ is the flat [NC][hc*H] mHC norm buffer: only the hc GEMVs read
  // it, and the MLA gather writes its per-token contexts to dsa_latent_.
  float* flat = latent_;
  const std::uint32_t flat_n = hc * h;

  float* cur = s0_;
  float* next = s1_;
  if (n == 1) {
    Dbg("hc_init-%u", 0U, s0_, sp);
  }
  std::uint32_t kda_slot = 0;
  std::uint32_t dsa_slot = 0;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    dbg_il_ = il;
    const DeviceLayer& l = model_->layers()[il];
    for (int block = 0; block < 2; ++block) {
      const bool attn = block == 0;
      const DeviceTensor& fn = attn ? l.hc_attn_fn : l.hc_ffn_fn;
      const DeviceTensor& base = attn ? l.hc_attn_base : l.hc_ffn_base;
      const DeviceTensor& scale = attn ? l.hc_attn_scale : l.hc_ffn_scale;
      const DeviceTensor& norm = attn ? l.attn_norm : l.ffn_norm;
      RmsNorm(cur, w_, flat, n, flat_n, flat_n, flat_n, c.rms_eps, nullptr);
      GemvN(n, fn, flat, flat_n, mix_a_, fn.rows);
      HcCoeffs(mix_a_, scale.f32(), base.f32(), c.hc_sinkhorn_iters, c.hc_eps,
               pre_a_, post_a_, comb_a_, n, nullptr);
      HcInputs(cur, pre_a_, hc, hc, h, x_, n, nullptr);
      RmsNorm(x_, norm.f32(), x_, n, h, h, h, c.rms_eps, nullptr);
      if (attn) {
        if (l.dsa) {
          DsaLayer(l, dsa_slot++, x_, n, p0, session, y0_);
        } else {
          KdaLayer(l, kda_slot++, x_, n, p0, session, y0_);
        }
      } else {
        Ffn(l, il, x_, n, y0_);
      }
      HcWriteback(y0_, cur, post_a_, comb_a_, hc, h, next, n, nullptr);
      std::swap(cur, next);
      if (n == 1) {
        Dbg(attn ? "hc_mixes_attn-%u" : "hc_mixes_ffn-%u", il, mix_a_,
            fn.rows);
        Dbg(attn ? "hc_pre-%u" : "hc_pre_ffn-%u", il, pre_a_, hc);
        Dbg(attn ? "hc_post-%u" : "hc_post_ffn-%u", il, post_a_, hc);
        Dbg(attn ? "hc_comb-%u" : "hc_comb_ffn-%u", il, comb_a_, hc * hc);
        Dbg(attn ? "attn_norm-%u" : "ffn_norm-%u", il, x_, h);
        Dbg(attn ? "hc_attn_post-%u" : "l_out-%u", il, cur, sp);
      }
    }
  }
  dbg_il_ = 0;

  HcInputs(cur, mean4_, 0U, hc, h, h_, n, nullptr);
  RmsNorm(h_, model_->output_norm().f32(), h_, n, h, h, h, c.rms_eps,
          nullptr);
  if (n == 1) {
    Dbg("hc_head-%u", 0U, h_, h);
    Dbg("result_norm-%u", 0U, h_, h);
  }

  // Only the last token of the run needs logits (the sampling point; the
  // earlier tokens' rows of h_ are prefill scratch).
  Gemv(model_->output().data, OfType(model_->output().type),
       model_->output().rows, h, model_->output().row_bytes,
       h_ + static_cast<std::size_t>(n - 1U) * h, logits_, nullptr);
  if (n == 1) {
    Dbg("result_output-%u", 0U, logits_, model_->output().rows);
  }
  session.pos_ = p0 + static_cast<std::int64_t>(n) - 1;
}

}  // namespace gufo::models::glm5_next::rocm
