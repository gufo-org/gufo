#include "src/models/qwen38_flash_next/reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

#include "src/models/qwen38_flash_next/cpu_ops.hpp"

namespace gufo::models::qwen38_flash_next {

using cpu::Sigmoid;
using cpu::Silu;

ReferenceModel::ReferenceModel(const ModelWeights& weights, NgramTable* ngram,
                               std::uint32_t max_context, Storage storage)
    : w_(weights),
      c_(weights.config),
      yarn_(MakeYarnRope(c_.rope_scaling, c_.rotary_dim, c_.rope_theta)),
      storage_(storage),
      ngram_(ngram),
      max_context_(max_context) {
  linear_.resize(c_.num_layers);
  attention_.resize(c_.num_layers);
  Reset();
}

namespace {
float Half(float value) {
  // The supported Linux x86-64 compiler implements IEEE binary16 conversion.
  return static_cast<float>(static_cast<_Float16>(value));
}

std::vector<float> QuantizedInput(std::span<const float> input) {
  std::vector<float> result(input.size());
  for (std::size_t base = 0; base < input.size(); base += 32) {
    float peak = 0;
    for (std::size_t j = 0; j < 32; ++j)
      peak = std::max(peak, std::abs(input[base + j]));
    const float scale = peak / 127.0F, stored = Half(scale);
    for (std::size_t j = 0; j < 32; ++j)
      result[base + j] =
          peak == 0 ? 0 : stored * std::round(input[base + j] / scale);
  }
  return result;
}
}  // namespace

void ReferenceModel::MatVec(const TensorRef& weight, std::uint64_t expert,
                            std::span<const float> x, std::span<float> out,
                            bool narrow_weights) {
  if (storage_ == Storage::kFloat32) {
    cpu::MatVec(weight, expert, x, out);
    return;
  }
  const auto quantized = weight.type == core::GgmlType::kQ8_0
                             ? QuantizedInput(x)
                             : std::vector<float>{};
  if (!quantized.empty())
    x = quantized;
  std::vector<float> row(weight.cols);
  for (std::size_t r = 0; r < weight.rows; ++r) {
    cpu::DequantizeRow(weight, expert, r, row.data());
    double sum = 0;
    for (std::size_t j = 0; j < x.size(); ++j)
      sum += double(narrow_weights ? Half(row[j]) : row[j]) * x[j];
    out[r] = static_cast<float>(sum);
  }
}

void ReferenceModel::Reset() {
  position_ = 0;
  mtp_position_ = 0;
  mtp_attention_ = {};
  mtp_hidden_.clear();
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    if (c_.IsLinearLayer(il)) {
      auto& s = linear_[il];
      s.conv.assign(static_cast<std::size_t>(c_.ssm_conv_kernel - 1) *
                        c_.SsmConvChannels(),
                    0.0F);
      s.state.assign(static_cast<std::size_t>(c_.ssm_num_v_heads) *
                         c_.ssm_head_dim * c_.ssm_head_dim,
                     0.0F);
    } else {
      auto& s = attention_[il];
      s.k.clear();
      s.v.clear();
      s.index_k.clear();
      s.block_k.clear();
      s.blocks = 0;
    }
  }
  ngram_history_.Reset();
  ple_conv_history_.assign(
      static_cast<std::size_t>(c_.PleConvHistory()) * c_.HcDim(), 0.0F);
}

// Grouped RMSNorm: each of the hc streams is normalized over hidden_size on
// its own, then the whole [hc_dim] gamma applies.
static void GroupedNorm(const Config& c, std::span<float> x, const float* w) {
  for (std::uint32_t s = 0; s < c.hc_count; ++s) {
    cpu::RmsNorm(
        x.subspan(static_cast<std::size_t>(s) * c.hidden_size, c.hidden_size),
        w + static_cast<std::size_t>(s) * c.hidden_size, c.rms_eps);
  }
}

void ReferenceModel::HcMix(const HcMixer& m, std::span<const float> res,
                           std::span<float> mixed, std::span<float> inject) {
  const std::uint32_t hc_dim = c_.HcDim();
  std::vector<float> xn(res.begin(), res.end());
  GroupedNorm(c_, xn, static_cast<const float*>(m.norm.data));

  std::vector<float> lo(c_.hc_low_rank);
  MatVec(m.down, 0, xn, lo);
  for (float& v : lo) {
    v = Silu(v / static_cast<float>(c_.hc_count));
  }
  std::vector<float> gate(hc_dim);
  MatVec(m.up, 0, lo, gate);

  std::fill(mixed.begin(), mixed.end(), 0.0F);
  for (std::uint32_t s = 0; s < c_.hc_count; ++s) {
    const std::size_t base = static_cast<std::size_t>(s) * c_.hidden_size;
    for (std::uint32_t i = 0; i < c_.hidden_size; ++i) {
      mixed[i] += xn[base + i] * Sigmoid(gate[base + i]);
    }
  }
  for (float& v : mixed) {
    v /= static_cast<float>(c_.hc_count);
  }
  if (!inject.empty()) {
    MatVec(m.inject, 0, xn, inject);
  }
}

void ReferenceModel::HcCombine(std::span<float> res,
                               std::span<const float> block_out,
                               std::span<const float> inject) {
  // 2*sigmoid centres the scatter weights on 1: zero injection logits make a
  // plain residual add into every stream.
  for (std::uint32_t s = 0; s < c_.hc_count; ++s) {
    const float w = 2.0F * Sigmoid(inject[s] / static_cast<float>(c_.hc_count));
    float* dst = res.data() + static_cast<std::size_t>(s) * c_.hidden_size;
    for (std::uint32_t i = 0; i < c_.hidden_size; ++i) {
      dst[i] += block_out[i] * w;
    }
  }
}

bool ReferenceModel::Ple(const LayerWeights& l, std::int32_t token,
                         std::span<float> res, std::string* error_msg) {
  const std::uint32_t hc_dim = c_.HcDim();
  std::vector<std::uint32_t> rows(c_.ple_heads);
  HashNgramRows(c_, ngram_history_, std::span<const std::int32_t>(&token, 1),
                rows);
  std::vector<float> emb(c_.PleEmbeddingDim());
  if (ngram_ == nullptr || !ngram_->Read(rows, emb)) {
    if (error_msg != nullptr) {
      *error_msg = "n-gram table read failed";
    }
    return false;
  }

  std::vector<float> key(hc_dim);
  std::vector<float> value(c_.hidden_size);
  MatVec(l.ple_key, 0, emb, key);
  MatVec(l.ple_value, 0, emb, value);
  GroupedNorm(c_, key, static_cast<const float*>(l.ple_norm_key.data));
  std::vector<float> query(res.begin(), res.end());
  GroupedNorm(c_, query, static_cast<const float*>(l.ple_norm_query.data));

  // Per-stream similarity, signed square root, sigmoid gate on the value.
  std::vector<float> gated(hc_dim);
  for (std::uint32_t s = 0; s < c_.hc_count; ++s) {
    const std::size_t base = static_cast<std::size_t>(s) * c_.hidden_size;
    double dot = 0.0;
    for (std::uint32_t i = 0; i < c_.hidden_size; ++i) {
      dot += static_cast<double>(key[base + i]) * query[base + i];
    }
    const float sc =
        static_cast<float>(dot) / std::sqrt(static_cast<float>(c_.hidden_size));
    const float mag = std::sqrt(std::max(std::fabs(sc), 1e-6F));
    const float gate =
        Sigmoid((sc < 0.0F ? -1.0F : (sc > 0.0F ? 1.0F : 0.0F)) * mag);
    for (std::uint32_t i = 0; i < c_.hidden_size; ++i) {
      gated[base + i] = value[i] * gate;
    }
  }

  std::vector<float> normalized(gated);
  GroupedNorm(c_, normalized, static_cast<const float*>(l.ple_norm_conv.data));

  // Depthwise causal conv over tokens, dilated by the n-gram size: tap k
  // reads (kernel-1-k)*dilation tokens back. History holds the previous
  // PleConvHistory() normalized vectors, oldest first.
  const std::uint32_t hist = c_.PleConvHistory();
  const std::uint32_t kern = c_.ple_conv_kernel;
  const std::uint32_t dil = c_.ple_ngram_size;
  const auto* conv_w = static_cast<const float*>(l.ple_conv1d.data);
  std::vector<float> conv_out(hc_dim, 0.0F);
  for (std::uint32_t k = 0; k < kern; ++k) {
    const std::uint32_t back = (kern - 1 - k) * dil;
    const float* src = back == 0
                           ? normalized.data()
                           : ple_conv_history_.data() +
                                 static_cast<std::size_t>(hist - back) * hc_dim;
    for (std::uint32_t ch = 0; ch < hc_dim; ++ch) {
      conv_out[ch] += conv_w[static_cast<std::size_t>(ch) * kern + k] * src[ch];
    }
  }
  // Shift history and append the current normalized vector.
  if (hist > 0) {
    std::copy(ple_conv_history_.begin() + hc_dim, ple_conv_history_.end(),
              ple_conv_history_.begin());
    std::copy(normalized.begin(), normalized.end(),
              ple_conv_history_.end() - hc_dim);
  }
  for (std::uint32_t ch = 0; ch < hc_dim; ++ch) {
    res[ch] += gated[ch] + Silu(conv_out[ch]);
  }
  return true;
}

void ReferenceModel::LinearAttention(const LayerWeights& l, LinearState& s,
                                     std::span<const float> x,
                                     std::span<float> out) {
  const std::uint32_t channels = c_.SsmConvChannels();
  const std::uint32_t key_dim = c_.SsmKeyDim();
  const std::uint32_t value_dim = c_.SsmValueDim();
  const std::uint32_t d = c_.ssm_head_dim;
  const std::uint32_t kern = c_.ssm_conv_kernel;

  std::vector<float> qkv(channels);
  std::vector<float> z(value_dim);
  std::vector<float> alpha(c_.ssm_num_v_heads);
  std::vector<float> beta(c_.ssm_num_v_heads);
  MatVec(l.ssm_qkv, 0, x, qkv);
  MatVec(l.ssm_gate, 0, x, z);
  MatVec(l.ssm_alpha, 0, x, alpha);
  MatVec(l.ssm_beta, 0, x, beta);

  // Causal depthwise conv over the last `kern` projections, then SiLU.
  const auto* conv_w = static_cast<const float*>(l.ssm_conv1d.data);
  std::vector<float> conv(channels);
  for (std::uint32_t ch = 0; ch < channels; ++ch) {
    float acc =
        conv_w[static_cast<std::size_t>(ch) * kern + kern - 1] * qkv[ch];
    for (std::uint32_t k = 0; k + 1 < kern; ++k) {
      acc += conv_w[static_cast<std::size_t>(ch) * kern + k] *
             s.conv[static_cast<std::size_t>(k) * channels + ch];
    }
    conv[ch] = Silu(acc);
  }
  if (kern > 1) {
    std::copy(s.conv.begin() + channels, s.conv.end(), s.conv.begin());
    std::copy(qkv.begin(), qkv.end(), s.conv.end() - channels);
  }

  float* q = conv.data();
  float* k = conv.data() + key_dim;
  float* v = conv.data() + 2 * key_dim;
  for (std::uint32_t h = 0; h < c_.ssm_num_k_heads; ++h) {
    cpu::L2Norm(std::span<float>(q + static_cast<std::size_t>(h) * d, d),
                c_.rms_eps);
    cpu::L2Norm(std::span<float>(k + static_cast<std::size_t>(h) * d, d),
                c_.rms_eps);
  }

  const auto* a = static_cast<const float*>(l.ssm_a.data);
  const auto* dt = static_cast<const float*>(l.ssm_dt.data);
  const auto* norm_w = static_cast<const float*>(l.ssm_norm.data);
  const float q_scale = 1.0F / std::sqrt(static_cast<float>(d));
  std::vector<float> attn(value_dim);
  std::vector<float> u(d);
  for (std::uint32_t h = 0; h < c_.ssm_num_v_heads; ++h) {
    // Value head h reads key head h % n_k_heads, the tiled layout the GGUF
    // converter permuted the value-side tensors into.
    const float* qh = q + static_cast<std::size_t>(h % c_.ssm_num_k_heads) * d;
    const float* kh = k + static_cast<std::size_t>(h % c_.ssm_num_k_heads) * d;
    const float* vh = v + static_cast<std::size_t>(h) * d;
    float* S = s.state.data() + static_cast<std::size_t>(h) * d * d;
    const float decay = std::exp(a[h] * cpu::Softplus(alpha[h] + dt[h]));
    const float b = Sigmoid(beta[h]);
    // S[j][i]: j over value dim, i over key dim.
    for (std::uint32_t j = 0; j < d; ++j) {
      float* row = S + static_cast<std::size_t>(j) * d;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < d; ++i) {
        row[i] *= decay;
        acc += static_cast<double>(row[i]) * kh[i];
      }
      u[j] = static_cast<float>(acc);
    }
    float* o = attn.data() + static_cast<std::size_t>(h) * d;
    for (std::uint32_t j = 0; j < d; ++j) {
      const float delta = (vh[j] - u[j]) * b;
      float* row = S + static_cast<std::size_t>(j) * d;
      double acc = 0.0;
      for (std::uint32_t i = 0; i < d; ++i) {
        row[i] += delta * kh[i];
        acc += static_cast<double>(row[i]) * qh[i];
      }
      o[j] = static_cast<float>(acc) * q_scale;
    }
    // Per-head RMSNorm, then the sigmoid output gate (Qwen3.5 used SiLU).
    cpu::RmsNorm(std::span<float>(o, d), norm_w, c_.rms_eps);
    for (std::uint32_t j = 0; j < d; ++j) {
      o[j] *= Sigmoid(z[static_cast<std::size_t>(h) * d + j]);
    }
  }
  MatVec(l.ssm_out, 0, attn, out);
}

void ReferenceModel::Attention(const LayerWeights& l, AttentionState& s,
                               std::span<const float> x, std::uint32_t pos,
                               std::span<float> out,
                               const qwen::vision::RopeLayout* layout) {
  const std::uint32_t hd = c_.head_dim;
  const std::uint32_t nh = c_.num_heads;
  const std::uint32_t nkv = c_.num_kv_heads;
  const std::uint32_t group = nh / nkv;
  const std::uint32_t idim = c_.indexer_head_dim;

  std::vector<float> qg(2 * c_.AttentionQDim());
  std::vector<float> k(c_.AttentionKvDim());
  std::vector<float> v(c_.AttentionKvDim());
  MatVec(l.attn_q, 0, x, qg);
  MatVec(l.attn_k, 0, x, k);
  MatVec(l.attn_v, 0, x, v);

  // wq interleaves [q | gate] per head.
  std::vector<float> q(c_.AttentionQDim());
  std::vector<float> gate(c_.AttentionQDim());
  for (std::uint32_t h = 0; h < nh; ++h) {
    std::copy_n(qg.begin() + static_cast<std::size_t>(h) * 2 * hd, hd,
                q.begin() + static_cast<std::size_t>(h) * hd);
    std::copy_n(qg.begin() + static_cast<std::size_t>(h) * 2 * hd + hd, hd,
                gate.begin() + static_cast<std::size_t>(h) * hd);
    cpu::RmsNorm(
        std::span<float>(q.data() + static_cast<std::size_t>(h) * hd, hd),
        static_cast<const float*>(l.attn_q_norm.data), c_.rms_eps);
  }
  for (std::uint32_t h = 0; h < nkv; ++h) {
    cpu::RmsNorm(
        std::span<float>(k.data() + static_cast<std::size_t>(h) * hd, hd),
        static_cast<const float*>(l.attn_k_norm.data), c_.rms_eps);
  }
  const auto rope = [&](float* values, std::uint32_t heads, std::uint32_t dim,
                        std::uint32_t physical) {
    if (layout == nullptr) {
      cpu::Rope(values, heads, dim, c_.rotary_dim, physical, c_.rope_theta,
                yarn_);
      return;
    }
    const auto coordinates = layout->Position(physical);
    const auto pairs = c_.rotary_dim / 2;
    for (std::uint32_t h = 0; h < heads; ++h) {
      for (std::uint32_t i = 0; i < pairs; ++i) {
        const auto axis = i % 3 == 1 && i < 33   ? 1
                          : i % 3 == 2 && i < 30 ? 2
                                                 : 0;
        float freq = std::pow(c_.rope_theta, -2.0F * i / c_.rotary_dim);
        if (yarn_.enabled != 0)
          freq *= YarnPairMultiplier(yarn_, i);
        const float angle = coordinates[axis] * freq;
        float cs = std::cos(angle);
        float sn = std::sin(angle);
        if (yarn_.enabled != 0) {
          cs *= yarn_.mscale;
          sn *= yarn_.mscale;
        }
        const auto offset = static_cast<std::size_t>(h) * dim + i;
        const float a = values[offset], b = values[offset + pairs];
        values[offset] = a * cs - b * sn;
        values[offset + pairs] = a * sn + b * cs;
      }
    }
  };
  rope(q.data(), nh, hd, pos);
  rope(k.data(), nkv, hd, pos);
  if (storage_ == Storage::kDecode) {
    for (auto& value : k)
      value = Half(value);
    for (auto& value : v)
      value = Half(value);
  }
  s.k.insert(s.k.end(), k.begin(), k.end());
  s.v.insert(s.v.end(), v.begin(), v.end());

  // QSA indexer: raw keys are cached; blocks of `ratio` complete positions
  // are pooled, normalized and rotated at the block start once complete.
  std::vector<float> iq(c_.indexer_heads * idim);
  std::vector<float> ik(idim);
  MatVec(l.indexer_q, 0, x, iq);
  MatVec(l.indexer_k, 0, x, ik);
  s.index_k.insert(s.index_k.end(), ik.begin(), ik.end());
  for (std::uint32_t h = 0; h < c_.indexer_heads; ++h) {
    cpu::RmsNorm(
        std::span<float>(iq.data() + static_cast<std::size_t>(h) * idim, idim),
        static_cast<const float*>(l.indexer_q_norm.data), c_.rms_eps);
  }
  rope(iq.data(), c_.indexer_heads, idim, pos);

  const std::uint32_t n_kv = pos + 1;
  const std::uint32_t ratio = c_.compress_ratio;
  const std::uint32_t complete = n_kv / ratio;
  while (s.blocks < complete) {
    std::vector<float> pooled(idim, 0.0F);
    for (std::uint32_t t = 0; t < ratio; ++t) {
      const float* src =
          s.index_k.data() +
          (static_cast<std::size_t>(s.blocks) * ratio + t) * idim;
      for (std::uint32_t i = 0; i < idim; ++i) {
        pooled[i] += src[i] / static_cast<float>(ratio);
      }
    }
    cpu::RmsNorm(pooled, static_cast<const float*>(l.indexer_k_norm.data),
                 c_.rms_eps);
    rope(pooled.data(), 1, idim, s.blocks * ratio);
    if (storage_ == Storage::kDecode)
      for (auto& value : pooled)
        value = Half(value);
    s.block_k.insert(s.block_k.end(), pooled.begin(), pooled.end());
    ++s.blocks;
  }

  // Visible keys: every position below the budget; otherwise the top scoring
  // complete blocks plus the incomplete tail.
  std::vector<std::uint32_t> visible;
  const std::uint32_t block_budget = c_.indexer_top_k / ratio;
  if (complete <= block_budget) {
    visible.resize(n_kv);
    std::iota(visible.begin(), visible.end(), 0U);
  } else {
    std::vector<float> score(complete, 0.0F);
    for (std::uint32_t b = 0; b < complete; ++b) {
      const float* kb = s.block_k.data() + static_cast<std::size_t>(b) * idim;
      for (std::uint32_t h = 0; h < c_.indexer_heads; ++h) {
        const float* qh = iq.data() + static_cast<std::size_t>(h) * idim;
        double dot = 0.0;
        for (std::uint32_t i = 0; i < idim; ++i) {
          dot += static_cast<double>(qh[i]) * kb[i];
        }
        score[b] += std::max(0.0F, static_cast<float>(dot));
      }
    }
    std::vector<std::uint32_t> order(complete);
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(
        order.begin(), order.end(),
        [&](std::uint32_t a, std::uint32_t b) { return score[a] > score[b]; });
    order.resize(block_budget);
    std::sort(order.begin(), order.end());
    for (std::uint32_t b : order) {
      for (std::uint32_t t = 0; t < ratio; ++t) {
        visible.push_back(b * ratio + t);
      }
    }
    for (std::uint32_t p = complete * ratio; p < n_kv; ++p) {
      visible.push_back(p);
    }
  }

  const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
  std::vector<float> ctx(c_.AttentionQDim(), 0.0F);
  std::vector<float> scores(visible.size());
  for (std::uint32_t h = 0; h < nh; ++h) {
    const std::uint32_t kvh = h / group;
    const float* qh = q.data() + static_cast<std::size_t>(h) * hd;
    float max_score = -INFINITY;
    for (std::size_t j = 0; j < visible.size(); ++j) {
      const float* kj =
          s.k.data() + (static_cast<std::size_t>(visible[j]) * nkv + kvh) * hd;
      double dot = 0.0;
      for (std::uint32_t i = 0; i < hd; ++i) {
        dot += static_cast<double>(qh[i]) * kj[i];
      }
      scores[j] = static_cast<float>(dot) * scale;
      max_score = std::max(max_score, scores[j]);
    }
    double denom = 0.0;
    for (float& sc : scores) {
      sc = std::exp(sc - max_score);
      denom += sc;
    }
    float* oh = ctx.data() + static_cast<std::size_t>(h) * hd;
    for (std::size_t j = 0; j < visible.size(); ++j) {
      const float p = static_cast<float>(scores[j] / denom);
      const float* vj =
          s.v.data() + (static_cast<std::size_t>(visible[j]) * nkv + kvh) * hd;
      for (std::uint32_t i = 0; i < hd; ++i) {
        oh[i] += p * vj[i];
      }
    }
    for (std::uint32_t i = 0; i < hd; ++i) {
      oh[i] *= Sigmoid(gate[static_cast<std::size_t>(h) * hd + i]);
    }
  }
  MatVec(l.attn_out, 0, ctx, out);
}

void ReferenceModel::Moe(const LayerWeights& l, std::span<const float> x,
                         std::span<float> out) {
  std::vector<float> logits(c_.num_experts);
  MatVec(l.router, 0, x, logits, true);
  // Softmax over every expert, top-k of the probabilities, renormalized.
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  double denom = 0.0;
  for (float& v : logits) {
    v = std::exp(v - max_logit);
    denom += v;
  }
  for (float& v : logits) {
    v = static_cast<float>(v / denom);
  }
  std::vector<std::uint32_t> order(c_.num_experts);
  std::iota(order.begin(), order.end(), 0U);
  std::partial_sort(
      order.begin(), order.begin() + c_.num_experts_used, order.end(),
      [&](std::uint32_t a, std::uint32_t b) { return logits[a] > logits[b]; });
  float sum = 0.0F;
  for (std::uint32_t i = 0; i < c_.num_experts_used; ++i) {
    sum += logits[order[i]];
  }
  sum = std::max(sum, 6.103515625e-5F);

  std::fill(out.begin(), out.end(), 0.0F);
  std::vector<float> gate(c_.expert_ff);
  std::vector<float> up(c_.expert_ff);
  std::vector<float> down(c_.hidden_size);
  for (std::uint32_t i = 0; i < c_.num_experts_used; ++i) {
    const std::uint32_t e = order[i];
    const float weight = logits[e] / sum;
    MatVec(l.ffn_gate_exps, e, x, gate);
    MatVec(l.ffn_up_exps, e, x, up);
    for (std::uint32_t j = 0; j < c_.expert_ff; ++j) {
      gate[j] = Silu(gate[j]) * up[j];
    }
    MatVec(l.ffn_down_exps, e, gate, down);
    for (std::uint32_t j = 0; j < c_.hidden_size; ++j) {
      out[j] += weight * down[j];
    }
  }

  // Shared expert with its own scalar sigmoid gate.
  std::vector<float> sgate(c_.shared_expert_ff);
  std::vector<float> sup(c_.shared_expert_ff);
  MatVec(l.shexp_gate, 0, x, sgate);
  MatVec(l.shexp_up, 0, x, sup);
  for (std::uint32_t j = 0; j < c_.shared_expert_ff; ++j) {
    sgate[j] = Silu(sgate[j]) * sup[j];
  }
  MatVec(l.shexp_down, 0, sgate, down);
  std::array<float, 1> shared_gate{};
  MatVec(l.shexp_gate_inp, 0, x, shared_gate, true);
  const float sg = Sigmoid(shared_gate[0]);
  for (std::uint32_t j = 0; j < c_.hidden_size; ++j) {
    out[j] += sg * down[j];
  }
}

bool ReferenceModel::Step(std::int32_t token, std::span<float> logits,
                          std::span<float> hc_out, std::string* error_msg) {
  if (position_ >= max_context_) {
    if (error_msg != nullptr) {
      *error_msg = "reference context is full";
    }
    return false;
  }
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg = "token out of range";
    }
    return false;
  }
  const std::uint32_t hidden = c_.hidden_size;
  const std::uint32_t hc_dim = c_.HcDim();

  // The wide residual starts as hc identical copies of the embedding.
  std::vector<float> embd(hidden);
  cpu::DequantizeRow(w_.token_embd, 0, static_cast<std::uint64_t>(token),
                     embd.data());
  std::vector<float> res(hc_dim);
  for (std::uint32_t s = 0; s < c_.hc_count; ++s) {
    std::copy(embd.begin(), embd.end(),
              res.begin() + static_cast<std::size_t>(s) * hidden);
  }

  std::vector<float> mixed(hidden);
  std::vector<float> inject(c_.hc_count);
  std::vector<float> block_out(hidden);
  for (std::uint32_t il = 0; il < c_.num_layers; ++il) {
    const LayerWeights& l = w_.layers[il];
    if (c_.IsPleLayer(il) && !Ple(l, token, res, error_msg)) {
      return false;
    }
    HcMix(l.hc_attn, res, mixed, inject);
    if (l.linear) {
      LinearAttention(l, linear_[il], mixed, block_out);
    } else {
      Attention(l, attention_[il], mixed, position_, block_out);
    }
    HcCombine(res, block_out, inject);

    HcMix(l.hc_ffn, res, mixed, inject);
    Moe(l, mixed, block_out);
    HcCombine(res, block_out, inject);
  }
  if (!hc_out.empty()) {
    std::copy(res.begin(), res.end(), hc_out.begin());
  }
  if (!logits.empty()) {
    HcMix(w_.hc_head, res, mixed, {});
    MatVec(w_.output, 0, mixed, logits);
  }
  ++position_;
  return true;
}

bool ReferenceModel::MtpStep(const MtpWeights& mtp, std::int32_t token,
                             std::span<const float> hidden,
                             std::span<float> logits, MtpTrace* trace,
                             const qwen::vision::RopeLayout* rope,
                             std::string* error_msg,
                             const MtpTrace* stage_inputs) {
  const auto H = c_.hidden_size, D = c_.HcDim();
  if (hidden.empty())
    hidden = mtp_hidden_;
  if (token < 0 || static_cast<std::uint32_t>(token) >= c_.vocab_size ||
      hidden.size() != D || mtp_position_ >= max_context_ ||
      (trace && !trace->Valid(H, D)) ||
      (stage_inputs && (stage_inputs->fused.size() != D ||
                        stage_inputs->attention.size() != D ||
                        stage_inputs->hidden.size() != D)) ||
      (!logits.empty() && logits.size() != c_.vocab_size)) {
    if (error_msg)
      *error_msg = "invalid reference MTP input";
    return false;
  }
  const auto copy = [](std::span<const float> src, std::span<float> dst) {
    if (!dst.empty())
      std::copy(src.begin(), src.end(), dst.begin());
  };
  const auto& l = mtp.block;
  std::vector<float> embedding(H), norm(hidden.begin(), hidden.end());
  cpu::DequantizeRow(w_.token_embd, 0, token, embedding.data());
  cpu::RmsNorm(embedding, static_cast<const float*>(l.nextn_enorm.data),
               c_.rms_eps);
  cpu::RmsNorm(norm, static_cast<const float*>(l.nextn_hnorm.data), c_.rms_eps);
  if (trace)
    copy(norm, trace->normalized_hidden);
  if (storage_ == Storage::kDecode &&
      l.nextn_eh_proj.type == core::GgmlType::kQ8_0) {
    embedding = QuantizedInput(embedding);
    norm = QuantizedInput(norm);
  }
  // Read the original combined GGUF rows, independently of GPU repacking.
  // Separate double accumulators implement fc_embedding + fc_hidden.
  std::vector<float> res(D), row(2 * H);
  for (std::uint32_t r = 0; r < H; ++r) {
    cpu::DequantizeRow(l.nextn_eh_proj, 0, r, row.data());
    double ep = 0;
    for (std::uint32_t j = 0; j < H; ++j)
      ep += static_cast<double>(row[j]) * embedding[j];
    for (std::uint32_t s = 0; s < c_.hc_count; ++s) {
      double hp = 0;
      for (std::uint32_t j = 0; j < H; ++j)
        hp += static_cast<double>(row[H + j]) * norm[s * H + j];
      res[s * H + r] = static_cast<float>(ep) + static_cast<float>(hp);
    }
  }
  if (trace)
    copy(res, trace->fused);
  if (stage_inputs)
    res.assign(stage_inputs->fused.begin(), stage_inputs->fused.end());
  std::vector<float> mixed(H), inject(c_.hc_count), block(H);
  HcMix(l.hc_attn, res, mixed, inject);
  Attention(l, mtp_attention_, mixed, mtp_position_, block, rope);
  HcCombine(res, block, inject);
  if (trace)
    copy(res, trace->attention);
  if (stage_inputs)
    res.assign(stage_inputs->attention.begin(), stage_inputs->attention.end());
  HcMix(l.hc_ffn, res, mixed, inject);
  if (trace)
    copy(mixed, trace->ffn_input);
  Moe(l, mixed, block);
  if (trace)
    copy(block, trace->ffn_output);
  HcCombine(res, block, inject);
  mtp_hidden_ = res;
  if (trace)
    copy(res, trace->hidden);
  if (stage_inputs)
    res.assign(stage_inputs->hidden.begin(), stage_inputs->hidden.end());
  if (!logits.empty() || (trace && !trace->head.empty())) {
    HcMix(l.nextn_head, res, mixed, {});
    if (trace)
      copy(mixed, trace->head);
    if (!logits.empty())
      MatVec(w_.output, 0, mixed, logits);
  }
  ++mtp_position_;
  return true;
}

}  // namespace gufo::models::qwen38_flash_next
